/* SPDX-License-Identifier: Apache-2.0 */
//! Standard Prometheus process metrics for the serving process.
//!
//! Every source is read directly from procfs or the process's resource limit.
//! A source that cannot be read or parsed omits only the metrics that depend on
//! it, so a scrape never fails and never panics on a missing or malformed file.

/// Appends the `process_*` metrics. Outside Linux the function writes nothing.
pub(crate) fn render(text: &mut String) {
    #[cfg(target_os = "linux")]
    render_linux(text);
    #[cfg(not(target_os = "linux"))]
    let _ = text;
}

/// Values parsed from `/proc/self/stat`: `utime`, `stime`, `starttime`, `vsize`,
/// and `rss` in the kernel's clock ticks, clock ticks since boot, bytes, and pages.
#[cfg(target_os = "linux")]
fn parse_stat(line: &str) -> Option<[u64; 5]> {
    // The command name is parenthesized and may itself contain `)` and spaces, so
    // the fields begin after the last `)`.
    let tail = line.get(line.rfind(')')? + 1..)?;
    let mut fields = tail.split_whitespace();
    // The state character follows the command name; rejecting a line without one
    // keeps this parser from reading unrelated text that happens to contain `)`.
    if !fields.next()?.chars().next().is_some_and(|state| "RSDZTtWXxKPI".contains(state)) {
        return None;
    }
    // Field 3 (state) is the first token of the tail; stat field N is token N - 3.
    let mut value = |skip: usize| -> Option<u64> { fields.nth(skip)?.parse().ok() };
    let utime = value(10)?;
    let stime = value(0)?;
    let starttime = value(6)?;
    let vsize = value(0)?;
    let rss = value(0)?;
    Some([utime, stime, starttime, vsize, rss])
}

#[cfg(target_os = "linux")]
fn parse_btime(contents: &str) -> Option<u64> {
    contents
        .lines()
        .find_map(|line| line.strip_prefix("btime "))
        .and_then(|value| value.trim().parse().ok())
}

#[cfg(target_os = "linux")]
fn metric(text: &mut String, name: &str, kind: &str, help: &str, value: impl std::fmt::Display) {
    use std::fmt::Write;
    let _ = writeln!(
        text,
        "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {value}"
    );
}

#[cfg(target_os = "linux")]
fn render_linux(text: &mut String) {
    // SAFETY: sysconf has no preconditions.
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    let ticks = (ticks > 0).then_some(ticks as u64);
    let page = crate::os::page_size() as u64;
    let stat = std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|line| parse_stat(&line));
    let btime = std::fs::read_to_string("/proc/stat")
        .ok()
        .and_then(|contents| parse_btime(&contents));
    let open_fds = crate::runtime_core::open_descriptors();
    // A zero target only reads the current soft limit.
    let max_fds = crate::os::raise_open_file_limit(0);

    if let (Some(stat), Some(ticks)) = (stat, ticks) {
        metric(
            text,
            "process_cpu_seconds_total",
            "counter",
            "User and system CPU time consumed by the serving process.",
            (stat[0] + stat[1]) as f64 / ticks as f64,
        );
    }
    if let Some(stat) = stat {
        metric(
            text,
            "process_virtual_memory_bytes",
            "gauge",
            "Virtual memory size of the serving process.",
            stat[3],
        );
    }
    if let Some(stat) = stat {
        metric(
            text,
            "process_resident_memory_bytes",
            "gauge",
            "Resident memory size of the serving process.",
            stat[4] * page,
        );
    }
    if let (Some(stat), Some(btime), Some(ticks)) = (stat, btime, ticks) {
        metric(
            text,
            "process_start_time_seconds",
            "gauge",
            "Start time of the serving process since the Unix epoch.",
            btime as f64 + stat[2] as f64 / ticks as f64,
        );
    }
    if let Some(open_fds) = open_fds {
        metric(
            text,
            "process_open_fds",
            "gauge",
            "Open file descriptors in the serving process, including the directory read that counts them.",
            open_fds,
        );
    }
    if let Some(max_fds) = max_fds {
        metric(
            text,
            "process_max_fds",
            "gauge",
            "Maximum open file descriptors for the serving process.",
            max_fds,
        );
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    fn value(text: &str, name: &str) -> f64 {
        let prefix = format!("{name} ");
        text.lines()
            .find_map(|line| line.strip_prefix(&prefix))
            .unwrap_or_else(|| panic!("missing {name}"))
            .parse()
            .unwrap()
    }

    #[test]
    fn parse_stat_reads_fields_after_a_parenthesized_command_with_a_paren() {
        assert_eq!(
            parse_stat("42 (a) b) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21"),
            Some([11, 12, 19, 20, 21])
        );
        assert_eq!(parse_stat("42 (no tail)"), None);
    }

    #[test]
    fn render_reports_process_memory_descriptors_and_start_time() {
        let mut text = String::new();
        render(&mut text);
        let resident = value(&text, "process_resident_memory_bytes");
        assert!(resident > 0.0, "{resident}");
        assert!(value(&text, "process_virtual_memory_bytes") >= resident);
        assert!(value(&text, "process_max_fds") >= value(&text, "process_open_fds"));
        assert!(value(&text, "process_open_fds") >= 1.0);
        let start = value(&text, "process_start_time_seconds");
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        assert!(start > now - 3600.0, "{start} {now}");
        assert!(start <= now + 1.0, "{start} {now}");
        for name in [
            "process_cpu_seconds_total",
            "process_resident_memory_bytes",
            "process_virtual_memory_bytes",
            "process_start_time_seconds",
            "process_open_fds",
            "process_max_fds",
        ] {
            assert!(text.contains(&format!("# TYPE {name} ")), "missing {name}");
        }
    }
}
