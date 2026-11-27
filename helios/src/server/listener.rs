/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use axum::serve::Listener;
use std::{
    io,
    net::SocketAddr,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    net::{TcpListener, TcpStream},
    sync::{OwnedSemaphorePermit, Semaphore},
};

#[cfg(test)]
const DEFAULT_MAX_CONNECTIONS: usize = 256;

pub(super) struct TrackedStream {
    inner: TcpStream,
    _permit: OwnedSemaphorePermit,
}

impl TrackedStream {
    #[cfg(test)]
    pub(super) fn nodelay(&self) -> io::Result<bool> {
        self.inner.nodelay()
    }
}

impl AsyncRead for TrackedStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for TrackedStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

pub(super) struct HttpListener {
    listener: TcpListener,
    log_to_stderr: bool,
    connections: Arc<Semaphore>,
}

impl HttpListener {
    #[cfg(test)]
    pub(super) fn new(listener: TcpListener) -> Self {
        Self::with_limits(listener, true, DEFAULT_MAX_CONNECTIONS)
    }

    pub(super) fn with_limits(
        listener: TcpListener,
        log_to_stderr: bool,
        max_connections: usize,
    ) -> Self {
        Self {
            listener,
            log_to_stderr,
            connections: Arc::new(Semaphore::new(max_connections.max(1))),
        }
    }
}

impl Listener for HttpListener {
    type Io = TrackedStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            let permit = self
                .connections
                .clone()
                .acquire_owned()
                .await
                .expect("HTTP connection semaphore is never closed while listening");
            let (stream, peer) = Listener::accept(&mut self.listener).await;
            match stream.set_nodelay(true) {
                Ok(()) => {
                    return (
                        TrackedStream {
                            inner: stream,
                            _permit: permit,
                        },
                        peer,
                    );
                }
                Err(error) => {
                    if self.log_to_stderr {
                        eprintln!(
                            "rejecting HTTP connection from {peer}: cannot set TCP_NODELAY: {error}"
                        );
                    }
                }
            }
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.listener.local_addr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn every_accepted_socket_disables_nagle() {
        tokio::time::timeout(Duration::from_secs(5), async {
            let tcp = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = tcp.local_addr().unwrap();
            let mut listener = HttpListener::new(tcp);
            assert_eq!(listener.local_addr().unwrap(), address);
            for _ in 0..2 {
                let client = TcpStream::connect(address).await.unwrap();
                client.set_nodelay(false).unwrap();
                let (accepted, peer) = listener.accept().await;
                assert_eq!(peer, client.local_addr().unwrap());
                assert!(accepted.nodelay().unwrap());
                assert!(!client.nodelay().unwrap());
            }
        })
        .await
        .expect("loopback socket acceptance timed out");
    }

    #[tokio::test]
    async fn connection_limit_blocks_excess_accept_until_release() {
        let tcp = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = tcp.local_addr().unwrap();
        let mut listener = HttpListener::with_limits(tcp, false, 1);
        let _c1 = TcpStream::connect(address).await.unwrap();
        let (accepted1, _) = listener.accept().await;
        let _c2 = TcpStream::connect(address).await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(50), listener.accept())
                .await
                .is_err(),
            "second accept must wait while the single connection permit is held"
        );
        drop(accepted1);
        let (_accepted2, _) = tokio::time::timeout(Duration::from_secs(2), listener.accept())
            .await
            .expect("second accept should proceed once the first stream drops");
    }
}
