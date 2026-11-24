/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Striped HTTP/2 and HTTP/1.1 client for Hub metadata and core content transfers.
//!
//! TLS connections negotiate `h2` first via ALPN and fall back to `http/1.1` when
//! the peer does not offer HTTP/2. Redirects are never followed automatically so
//! the caller controls hop bounds and same-origin credential isolation.

use std::{
    future::Future,
    io,
    pin::Pin,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};

use bytes::Bytes;
use http::{HeaderMap, Request, Uri, Version};
use http_body_util::{BodyExt, Empty};
use hyper::body::Incoming;
use hyper_util::{
    client::legacy::{
        Client,
        connect::{Connected, Connection, HttpConnector},
    },
    rt::{TokioExecutor, TokioIo, TokioTimer},
};
use rustls::{ClientConfig, RootCertStore};
use rustls_pki_types::ServerName;
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;
use tower_service::Service;

/// Maximum poll interval while waiting for the next body frame so cancellation
/// aborts stalled streams promptly without weakening the total body deadline.
const READ_POLL_QUANTUM: Duration = Duration::from_millis(200);

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Transport error returned by `HttpClient`. Never embeds raw request URLs so
/// presigned query parameters or credentials cannot leak into logs or messages.
#[derive(Debug)]
pub enum HttpError {
    InvalidUri,
    InvalidHeader(String),
    Timeout(String),
    Transport(String),
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidUri => write!(f, "invalid request URL"),
            Self::InvalidHeader(msg) => write!(f, "invalid request header: {msg}"),
            Self::Timeout(msg) => write!(f, "{msg}"),
            Self::Transport(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for HttpError {}

impl HttpError {
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Timeout(_) | Self::Transport(_))
    }
}

/// Connector supporting plaintext `http` (HTTP/1.1) and TLS `https` with ALPN
/// `["h2", "http/1.1"]`.
#[derive(Clone)]
struct HttpsConnector {
    http: HttpConnector,
    tls: TlsConnector,
    handshake_timeout: Duration,
}

fn shared_runtime() -> &'static Arc<tokio::runtime::Runtime> {
    static RUNTIME: OnceLock<Arc<tokio::runtime::Runtime>> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(8)
                .enable_io()
                .enable_time()
                .thread_name("flare-http")
                .build()
                .expect("HTTP client runtime initialization failed"),
        )
    })
}

fn shared_tls_config() -> &'static Arc<ClientConfig> {
    static TLS: OnceLock<Arc<ClientConfig>> = OnceLock::new();
    TLS.get_or_init(|| {
        let mut roots = RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let mut tls_config =
            ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
                .with_safe_default_protocol_versions()
                .expect("TLS default protocol versions")
                .with_root_certificates(roots)
                .with_no_client_auth();
        tls_config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
        Arc::new(tls_config)
    })
}

impl HttpsConnector {
    fn new(connect_timeout: Duration) -> Self {
        let mut http = HttpConnector::new();
        http.enforce_http(false);
        http.set_connect_timeout(Some(connect_timeout));
        http.set_nodelay(true);
        http.set_keepalive(Some(Duration::from_secs(30)));

        Self {
            http,
            tls: TlsConnector::from(Arc::clone(shared_tls_config())),
            handshake_timeout: connect_timeout,
        }
    }
}

enum MaybeTlsStream {
    Plain(TokioIo<TcpStream>),
    Tls(TokioIo<tokio_rustls::client::TlsStream<TcpStream>>),
}

impl hyper::rt::Read for MaybeTlsStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: hyper::rt::ReadBufCursor<'_>,
    ) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_read(cx, buf),
            Self::Tls(stream) => Pin::new(stream).poll_read(cx, buf),
        }
    }
}

impl hyper::rt::Write for MaybeTlsStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        match self.get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_write(cx, buf),
            Self::Tls(stream) => Pin::new(stream).poll_write(cx, buf),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_flush(cx),
            Self::Tls(stream) => Pin::new(stream).poll_flush(cx),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_shutdown(cx),
            Self::Tls(stream) => Pin::new(stream).poll_shutdown(cx),
        }
    }

    fn is_write_vectored(&self) -> bool {
        match self {
            Self::Plain(stream) => stream.is_write_vectored(),
            Self::Tls(stream) => stream.is_write_vectored(),
        }
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        match self.get_mut() {
            Self::Plain(stream) => Pin::new(stream).poll_write_vectored(cx, bufs),
            Self::Tls(stream) => Pin::new(stream).poll_write_vectored(cx, bufs),
        }
    }
}

impl Connection for MaybeTlsStream {
    fn connected(&self) -> Connected {
        match self {
            Self::Plain(stream) => stream.connected(),
            Self::Tls(stream) => {
                let (tcp, session) = stream.inner().get_ref();
                let mut connected = Connected::new();
                if let (Ok(remote), Ok(local)) = (tcp.peer_addr(), tcp.local_addr()) {
                    let _ = (remote, local);
                }
                if session.alpn_protocol() == Some(b"h2") {
                    connected = connected.negotiated_h2();
                }
                connected
            }
        }
    }
}

impl Service<Uri> for HttpsConnector {
    type Response = MaybeTlsStream;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.http.poll_ready(cx).map_err(Into::into)
    }

    fn call(&mut self, uri: Uri) -> Self::Future {
        let scheme = uri.scheme_str().unwrap_or("");
        let is_https = match scheme {
            "https" => true,
            "http" => false,
            other => {
                let message = format!("unsupported URL scheme: {other}");
                return Box::pin(async move { Err(io::Error::other(message).into()) });
            }
        };
        let host = uri
            .host()
            .map(|h| h.trim_start_matches('[').trim_end_matches(']').to_owned());
        let tcp_fut = self.http.call(uri);
        let tls = self.tls.clone();
        let handshake_timeout = self.handshake_timeout;

        Box::pin(async move {
            let tcp_io = tcp_fut.await.map_err(|error| -> BoxError { Box::new(error) })?;
            if !is_https {
                return Ok(MaybeTlsStream::Plain(tcp_io));
            }
            let host = host.ok_or_else(|| io::Error::other("HTTPS URL missing host"))?;
            let server_name = ServerName::try_from(host)
                .map_err(|_| io::Error::other("invalid TLS server name"))?;
            let tls_stream = tokio::time::timeout(
                handshake_timeout,
                tls.connect(server_name, tcp_io.into_inner()),
            )
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "TLS handshake timed out"))??;
            Ok(MaybeTlsStream::Tls(TokioIo::new(tls_stream)))
        })
    }
}

type HyperClient = Client<HttpsConnector, Empty<Bytes>>;

/// Striped HTTP/2 + HTTP/1.1 client safe to share across blocking worker threads.
///
/// Striping across `N` independent `hyper` pools lets concurrent range transfers
/// multiplex HTTP/2 streams across multiple TLS connections (and distinct CDN
/// edge IPs) instead of bottlenecking every stream onto one TCP flow.
pub struct HttpClient {
    runtime: Arc<tokio::runtime::Runtime>,
    clients: Vec<HyperClient>,
    next_stripe: AtomicUsize,
    response_timeout: Duration,
    body_timeout: Duration,
}

impl HttpClient {
    pub fn new(
        stripes: usize,
        connect_timeout: Duration,
        response_timeout: Duration,
        body_timeout: Duration,
    ) -> Self {
        let runtime = Arc::clone(shared_runtime());
        let stripe_count = stripes.max(1);
        let mut clients = Vec::with_capacity(stripe_count);
        for _ in 0..stripe_count {
            let connector = HttpsConnector::new(connect_timeout);
            let mut builder = Client::builder(TokioExecutor::new());
            // Do not enable `http2_adaptive_window`: hyper overrides explicit
            // stream and connection window sizes back to 64 KiB when enabled.
            builder
                .timer(TokioTimer::new())
                .pool_timer(TokioTimer::new())
                .pool_idle_timeout(Duration::from_secs(60))
                .pool_max_idle_per_host(16)
                .http2_initial_stream_window_size(Some(8 << 20))
                .http2_initial_connection_window_size(Some(64 << 20))
                .http2_max_frame_size(Some(256 << 10))
                .http2_keep_alive_interval(Some(Duration::from_secs(20)))
                .http2_keep_alive_timeout(Duration::from_secs(20))
                .http2_keep_alive_while_idle(true);
            clients.push(builder.build(connector));
        }
        Self {
            runtime,
            clients,
            next_stripe: AtomicUsize::new(0),
            response_timeout,
            body_timeout,
        }
    }

    /// Issue one GET request without following redirects.
    pub fn get(
        &self,
        url: &str,
        headers: &[(&str, &str)],
    ) -> Result<HttpResponse, HttpError> {
        let uri: Uri = url.parse().map_err(|_| HttpError::InvalidUri)?;
        if !matches!(uri.scheme_str(), Some("http" | "https")) || uri.authority().is_none() {
            return Err(HttpError::InvalidUri);
        }
        let mut builder = Request::builder()
            .method(http::Method::GET)
            .uri(uri)
            .header(http::header::USER_AGENT, "flare");
        for &(name, value) in headers {
            builder = builder.header(name, value);
        }
        let request = builder
            .body(Empty::<Bytes>::new())
            .map_err(|error| HttpError::InvalidHeader(error.to_string()))?;

        let stripe = self.next_stripe.fetch_add(1, Ordering::Relaxed) % self.clients.len();
        let client = self.clients[stripe].clone();
        let response_timeout = self.response_timeout;

        let response = self
            .runtime
            .block_on(async move { tokio::time::timeout(response_timeout, client.request(request)).await })
            .map_err(|_| HttpError::Timeout("HTTP response headers timed out".to_owned()))?
            .map_err(classify_hyper_error)?;

        let status = response.status().as_u16();
        let version = response.version();
        let headers = response.headers().clone();
        let body = response.into_body();
        Ok(HttpResponse {
            status,
            version,
            headers,
            body,
            runtime: Arc::clone(&self.runtime),
            body_timeout: self.body_timeout,
        })
    }
}

fn classify_hyper_error(error: hyper_util::client::legacy::Error) -> HttpError {
    if error.is_connect() {
        return HttpError::Transport("connection failed".to_owned());
    }
    if let Some(hyper_err) = std::error::Error::source(&error)
        .and_then(|source| source.downcast_ref::<hyper::Error>())
        && (hyper_err.is_closed()
            || hyper_err.is_incomplete_message()
            || hyper_err.is_canceled())
    {
        return HttpError::Transport("connection closed before response completed".to_owned());
    }
    let message = error.to_string();
    if message.to_ascii_lowercase().contains("timed out") {
        HttpError::Timeout("HTTP transport timed out".to_owned())
    } else {
        // Keep transport diagnostics free of any URL details.
        HttpError::Transport("HTTP transport error".to_owned())
    }
}

pub struct HttpResponse {
    status: u16,
    version: Version,
    headers: HeaderMap,
    body: Incoming,
    runtime: Arc<tokio::runtime::Runtime>,
    body_timeout: Duration,
}

impl HttpResponse {
    pub fn status(&self) -> u16 {
        self.status
    }

    pub fn version(&self) -> Version {
        self.version
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name)?.to_str().ok()
    }

    pub fn into_reader(self) -> HttpBodyReader {
        let deadline = tokio::time::Instant::now() + self.body_timeout;
        HttpBodyReader {
            body: self.body,
            pending: Bytes::new(),
            runtime: self.runtime,
            deadline,
            done: false,
        }
    }

    pub fn read_to_string_limit(self, limit: u64) -> Result<String, HttpError> {
        let mut reader = self.into_reader();
        let mut out = Vec::new();
        let mut scratch = [0u8; 16 << 10];
        loop {
            match reader.read_poll(&mut scratch, &|| false) {
                Ok(0) => break,
                Ok(n) => {
                    if out.len() as u64 + n as u64 > limit {
                        return Err(HttpError::Transport(format!(
                            "response body exceeded {limit} byte limit"
                        )));
                    }
                    out.extend_from_slice(&scratch[..n]);
                }
                Err(error) if error.kind() == io::ErrorKind::TimedOut => {
                    return Err(HttpError::Timeout(error.to_string()));
                }
                Err(error) => return Err(HttpError::Transport(error.to_string())),
            }
        }
        String::from_utf8(out).map_err(|_| HttpError::Transport("response body is not valid UTF-8".to_owned()))
    }
}

pub struct HttpBodyReader {
    body: Incoming,
    pending: Bytes,
    runtime: Arc<tokio::runtime::Runtime>,
    deadline: tokio::time::Instant,
    done: bool,
}

impl HttpBodyReader {
    /// Read the next bytes while checking `is_cancelled` periodically so stalled
    /// transfers abort quickly on cancellation or sibling error.
    pub fn read_poll(
        &mut self,
        buf: &mut [u8],
        is_cancelled: &impl Fn() -> bool,
    ) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        if !self.pending.is_empty() {
            let take = buf.len().min(self.pending.len());
            buf[..take].copy_from_slice(&self.pending[..take]);
            let _ = self.pending.split_to(take);
            return Ok(take);
        }
        if self.done {
            return Ok(0);
        }
        loop {
            if is_cancelled() {
                return Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"));
            }
            let now = tokio::time::Instant::now();
            if now >= self.deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "response body deadline exceeded",
                ));
            }
            let step_deadline = (now + READ_POLL_QUANTUM).min(self.deadline);
            let next = self
                .runtime
                .block_on(async { tokio::time::timeout_at(step_deadline, self.body.frame()).await });
            match next {
                Err(_) => {
                    if tokio::time::Instant::now() >= self.deadline {
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "response body deadline exceeded",
                        ));
                    }
                    continue;
                }
                Ok(None) => {
                    self.done = true;
                    return Ok(0);
                }
                Ok(Some(Err(error))) => {
                    return Err(io::Error::other(error.to_string()));
                }
                Ok(Some(Ok(frame))) => {
                    let Ok(mut data) = frame.into_data() else {
                        continue;
                    };
                    if data.is_empty() {
                        continue;
                    }
                    let take = buf.len().min(data.len());
                    buf[..take].copy_from_slice(&data[..take]);
                    let _ = data.split_to(take);
                    self.pending = data;
                    return Ok(take);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;

    #[test]
    fn http_client_rejects_invalid_and_unsupported_urls_without_echoing_them() {
        let client = HttpClient::new(
            2,
            Duration::from_secs(5),
            Duration::from_secs(5),
            Duration::from_secs(5),
        );
        let secret = "super-secret-query-token";
        for url in [
            format!("ftp://example.invalid/{secret}"),
            format!("https://user:{secret}@example.invalid/file with space"),
        ] {
            let err = client.get(&url, &[]).err().unwrap();
            assert!(!err.is_retryable());
            assert!(!err.to_string().contains(secret), "{err}");
        }
    }

    #[test]
    fn http_client_streams_body_and_aborts_promptly_when_cancelled() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut req_buf = [0u8; 1024];
            let _ = io::Read::read(&mut stream, &mut req_buf);
            let _ = stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\nhello",
            );
            std::thread::sleep(Duration::from_secs(5));
        });

        let client = HttpClient::new(
            2,
            Duration::from_secs(5),
            Duration::from_secs(5),
            Duration::from_secs(30),
        );
        let response = client.get(&format!("http://{addr}/slow"), &[]).unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(response.version(), Version::HTTP_11);
        let mut reader = response.into_reader();
        let mut buf = [0u8; 16];
        let n = reader.read_poll(&mut buf, &|| false).unwrap();
        assert_eq!(&buf[..n], b"hello");

        let cancelled = std::sync::atomic::AtomicBool::new(false);
        let start = std::time::Instant::now();
        let cancel_at = start + Duration::from_millis(120);
        let err = reader
            .read_poll(&mut buf, &|| {
                if std::time::Instant::now() >= cancel_at {
                    cancelled.store(true, Ordering::Release);
                }
                cancelled.load(Ordering::Acquire)
            })
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Interrupted);
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "cancellation must abort well before the 5s server stall"
        );
        drop(server);
    }
}
