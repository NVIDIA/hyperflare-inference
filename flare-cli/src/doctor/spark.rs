/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use std::{fs, io, net::Ipv4Addr, path::Path};

use flare_helios::{
    core::naming::platform_from_pci,
    execution::roce_fabric::{
        Fabric, MAX_DEVICES, has_rocev2_ipv4_gid, ipv4_subnet_mask, link_local, port1_active,
    },
};

use super::{Check, CheckStatus, parse_u64_param};

const CONNECTX_7: u32 = 0x1021_15b3;
const JUMBO_MTU: u32 = 9000;

fn pci_id(device: &Path) -> Option<u32> {
    let read_id = |name| u16::try_from(parse_u64_param(&read(&device.join(name))?)?).ok();
    // NVML packs the device ID in the upper half and the vendor ID in the lower half.
    Some((u32::from(read_id("device")?) << 16) | u32::from(read_id("vendor")?))
}

fn read(file: &Path) -> Option<String> {
    fs::read_to_string(file).ok().map(|s| s.trim().to_owned())
}

pub(super) fn check_host(
    sysfs: &Path,
    discover: impl FnOnce() -> flare_helios::Result<Fabric>,
) -> Option<Check> {
    let mut gb10 = false;
    let mut cx7 = false;
    for id in fs::read_dir(sysfs.join("bus/pci/devices"))
        .ok()?
        .filter_map(Result::ok)
        .filter_map(|device| pci_id(&device.path()))
    {
        gb10 |= platform_from_pci(id) == Some("gb10");
        cx7 |= id == CONNECTX_7;
    }
    if !gb10 {
        return None;
    }
    if !cx7 {
        return Some(result(
            CheckStatus::NotApplicable,
            Some("No ConnectX-7 PCI device".into()),
            "This host has no ConnectX-7 adapter.".into(),
            None,
        ));
    }
    Some(check_fabric(sysfs, discover))
}

fn check_fabric(sysfs: &Path, discover: impl FnOnce() -> flare_helios::Result<Fabric>) -> Check {
    match inspect(sysfs, discover) {
        Ok(outcome) => outcome,
        Err(error) => result(
            CheckStatus::Unverified,
            None,
            error.to_string(),
            Some("Make the host's RDMA sysfs attributes visible to flare doctor."),
        ),
    }
}

fn result(
    status: CheckStatus,
    actual: Option<String>,
    detail: String,
    remediation: Option<&str>,
) -> Check {
    Check {
        id: "spark.cx7".into(),
        name: "ConnectX-7 fabric".into(),
        status,
        expected: Some("An active CX7 link with an assigned IPv4 address and a RoCEv2 GID".into()),
        actual,
        detail,
        remediation: remediation.map(str::to_owned),
    }
}

fn inspect(
    sysfs: &Path,
    discover: impl FnOnce() -> flare_helios::Result<Fabric>,
) -> io::Result<Check> {
    let root = sysfs.join("class/infiniband");
    let mut devices = match fs::read_dir(&root) {
        Ok(entries) => entries.collect::<io::Result<Vec<_>>>()?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error),
    };
    devices.retain(|device| pci_id(&device.path().join("device")) == Some(CONNECTX_7));
    devices.sort_by_key(|device| device.file_name());
    if devices.is_empty() {
        return Ok(result(
            CheckStatus::Warn,
            Some("No ConnectX-7 RDMA devices".into()),
            "ConnectX-7 is present on PCI, but the kernel exposes no CX7 RDMA devices.".into(),
            Some(
                "Check that the ConnectX-7 adapter is visible and the mlx5 RDMA driver is loaded.",
            ),
        ));
    }

    let mut active = Vec::new();
    let mut unreadable = Vec::new();
    for device in &devices {
        let name = device.file_name().to_string_lossy().into_owned();
        match port1_active(&device.path()) {
            Ok(true) => active.push(name),
            Ok(false) => {}
            Err(_) => unreadable.push(name),
        }
    }
    if active.is_empty() {
        if !unreadable.is_empty() {
            return Ok(result(
                CheckStatus::Unverified,
                None,
                format!(
                    "Could not read the link state for {}.",
                    unreadable.join(", ")
                ),
                Some("Check the host's RDMA port state in /sys/class/infiniband."),
            ));
        }
        let names: Vec<_> = devices
            .iter()
            .map(|device| device.file_name().to_string_lossy().into_owned())
            .collect();
        return Ok(result(
            CheckStatus::Warn,
            Some("No active ConnectX-7 links".into()),
            format!(
                "No active ConnectX-7 RDMA links ({}) for multi-host serving.",
                names.join(", ")
            ),
            Some("Connect a CX7 port to a peer or switch and bring up its network interface."),
        ));
    }

    // Use the same GID index and assigned-address verification as serving.
    let fabric = discover();
    let mut ready = Vec::new();
    if let Ok(fabric) = &fabric {
        for endpoint in &fabric.endpoints {
            if !active.contains(&endpoint.name) {
                continue;
            }
            let Some((address, prefix)) = endpoint.ipv4 else {
                continue;
            };
            let netdev = root
                .join(&endpoint.name)
                .join(format!("ports/1/gid_attrs/ndevs/{}", fabric.gid_index));
            let Some(interface) = read(&netdev) else {
                continue;
            };
            let mtu = read(&sysfs.join("class/net").join(&interface).join("mtu"))
                .and_then(|mtu| mtu.parse().ok());
            ready.push(Interface {
                device: &endpoint.name,
                interface,
                address,
                prefix,
                mtu,
            });
        }
        if !ready.is_empty() {
            let listed = ready
                .iter()
                .map(Interface::describe)
                .collect::<Vec<_>>()
                .join(", ");
            let mut detail = format!(
                "{} of {} active CX7 devices supply IPv4 endpoints for serving at RoCEv2 GID index {}: {listed}.",
                ready.len(),
                active.len(),
                fabric.gid_index
            );
            let mut warnings = configuration_warnings(&ready);
            if ready.iter().any(Interface::is_link_local) {
                detail.push_str(" Link-local IPv4 requires an LLDP-confirmed direct peer.");
            }
            let mut omitted = Vec::new();
            let mut missing = Vec::new();
            for name in &active {
                if ready.iter().any(|port| port.device == name) {
                    continue;
                }
                if fabric.endpoints.len() == MAX_DEVICES
                    && !fabric
                        .endpoints
                        .iter()
                        .any(|endpoint| endpoint.name == *name)
                    && has_rocev2_ipv4_gid(&root.join(name), fabric.gid_index)
                {
                    omitted.push(name.as_str());
                } else {
                    missing.push(name.as_str());
                }
            }
            if !omitted.is_empty() {
                detail.push_str(&format!(
                    " Serving selects at most {MAX_DEVICES} devices and omitted {}.",
                    omitted.join(", ")
                ));
            }
            if !missing.is_empty() {
                warnings.push(format!(
                    "Check the IPv4 address and GID on {}.",
                    missing.join(", ")
                ));
            }
            if !unreadable.is_empty() {
                warnings.push(format!(
                    "Could not read the link state for {}.",
                    unreadable.join(", ")
                ));
            }
            if warnings.is_empty() {
                return Ok(result(CheckStatus::Pass, Some(listed), detail, None));
            }
            detail.push(' ');
            detail.push_str(&warnings.join(" "));
            return Ok(result(
                CheckStatus::Warn,
                Some(listed),
                detail,
                Some("Check IPv4 subnet and MTU settings against the intended peers and switch."),
            ));
        }
    }

    let reason = match fabric {
        Err(error) => error.to_string(),
        Ok(fabric) => format!(
            "no CX7 device has an assigned IPv4 address at RoCEv2 GID index {}",
            fabric.gid_index
        ),
    };
    Ok(result(
        CheckStatus::Warn,
        Some("No usable ConnectX-7 IPv4 endpoint".into()),
        format!("Active ConnectX-7 links ({}): {reason}.", active.join(", ")),
        Some("Assign IPv4 addresses on the CX7 interfaces and check their RoCEv2 GID entries."),
    ))
}

struct Interface<'a> {
    device: &'a str,
    interface: String,
    address: Ipv4Addr,
    prefix: u8,
    mtu: Option<u32>,
}

impl Interface<'_> {
    fn is_link_local(&self) -> bool {
        link_local((u32::from(self.address), self.prefix))
    }

    fn describe(&self) -> String {
        let mtu = self
            .mtu
            .map_or_else(String::new, |mtu| format!(", MTU {mtu}"));
        format!(
            "{} ({}, {}/{}{mtu})",
            self.device, self.interface, self.address, self.prefix
        )
    }
}

fn configuration_warnings(ports: &[Interface<'_>]) -> Vec<String> {
    let mut warnings = Vec::new();
    for (i, port) in ports.iter().enumerate() {
        let name = &port.interface;
        let address = u32::from(port.address);
        let host_bits = !ipv4_subnet_mask(port.prefix);
        let host = address & host_bits;
        if matches!(port.prefix, 0 | 32) || (port.prefix < 31 && (host == 0 || host == host_bits)) {
            warnings.push(format!(
                "Review {name}'s IPv4 subnet ({}/{}).",
                port.address, port.prefix
            ));
        }

        match port.mtu {
            Some(mtu) if mtu < JUMBO_MTU => {
                warnings.push(format!(
                    "{name}'s MTU {mtu} is below {JUMBO_MTU}. MTU {JUMBO_MTU} can improve throughput when the full path supports it."
                ));
            }
            None => warnings.push(format!("Could not read the MTU for {name}.")),
            _ => {}
        }

        for other in &ports[..i] {
            if port.interface == other.interface || (port.is_link_local() && other.is_link_local())
            {
                continue;
            }
            let mask = ipv4_subnet_mask(port.prefix.min(other.prefix));
            if address & mask == u32::from(other.address) & mask {
                warnings.push(format!(
                    "IPv4 subnets overlap on {} and {name}.",
                    other.interface
                ));
            }
        }
    }
    warnings
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use crate::doctor::{GpuProbe, Interconnect, Report, evaluate_linux_host, render};
    use crate::inspect::Canvas;
    use flare_helios::execution::roce_fabric;

    fn write(root: &Path, path: &str, content: &str) {
        let file = root.join(path);
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(file, content).unwrap();
    }

    fn add_spark(sysfs: &Path) {
        write(sysfs, "bus/pci/devices/000f:01:00.0/vendor", "0x10de\n");
        write(sysfs, "bus/pci/devices/000f:01:00.0/device", "0x2e12\n");
        write(sysfs, "bus/pci/devices/0000:01:00.0/vendor", "0x15b3\n");
        write(sysfs, "bus/pci/devices/0000:01:00.0/device", "0x1021\n");
    }

    fn report(sysfs: &Path, prefix: impl Fn(&str, Ipv4Addr) -> Option<u8>) -> Report {
        evaluate_linux_host(
            GpuProbe {
                gpus: Vec::new(),
                interconnect: Interconnect::SingleGpu,
                notes: Vec::new(),
            },
            sysfs,
            &sysfs.join("params"),
            &sysfs.join("cmdline"),
            || {
                roce_fabric::discover_in(&sysfs.join("class/infiniband"), prefix, |_| {
                    Default::default()
                })
            },
        )
    }

    #[test]
    fn cx7_readiness_requires_a_live_ipv4_address_at_the_serving_gid_index() {
        let dir = tempfile::tempdir().unwrap();
        let sysfs = dir.path();
        add_spark(sysfs);
        let no_driver = report(sysfs, |_, _| None);
        assert!(no_driver.ok);
        assert_eq!(no_driver.checks[1].status, CheckStatus::Warn);
        assert!(
            no_driver.checks[1]
                .detail
                .contains("kernel exposes no CX7 RDMA devices")
        );

        let root = sysfs.join("class/infiniband");
        let port = root.join("roce0/ports/1");
        write(&root, "roce0/device/vendor", "0x15b3\n");
        write(&root, "roce0/device/device", "0x1021\n");
        write(&root, "roce1/device/vendor", "0x15b3\n");
        write(&root, "roce1/device/device", "0x1021\n");
        write(&root, "roce1/ports/1/state", "1: DOWN\n");
        write(&port, "state", "1: DOWN\n");
        let no_links = report(sysfs, |_, _| None);
        assert!(no_links.ok);
        assert_eq!(no_links.checks[1].status, CheckStatus::Warn);
        assert!(
            no_links.checks[1]
                .detail
                .contains("No active ConnectX-7 RDMA links")
        );
        write(&port, "state", "4: ACTIVE\n");
        write(&port, "gid_attrs/ndevs/3", "enp1s0f0np0\n");
        write(&port, "gids/3", "0000:0000:0000:0000:0000:ffff:c0a8:0b02\n");
        write(sysfs, "class/net/enp1s0f0np0/mtu", "9000\n");

        let inspect = |has_ip| {
            report(sysfs, |netdev, addr| {
                (has_ip && netdev == "enp1s0f0np0" && addr.octets() == [192, 168, 11, 2])
                    .then_some(24)
            })
        };
        write(&port, "gid_attrs/types/3", "RoCE v1\n");
        assert_eq!(inspect(true).checks[1].status, CheckStatus::Warn);
        write(&port, "gid_attrs/types/3", "RoCE v2\n");
        assert_eq!(inspect(false).checks[1].status, CheckStatus::Warn);
        let report = inspect(true);
        let ready = &report.checks[1];
        assert!(report.ok);
        assert_eq!(ready.status, CheckStatus::Pass, "{}", ready.detail);
        assert!(
            ready
                .actual
                .as_deref()
                .unwrap()
                .contains("enp1s0f0np0, 192.168.11.2/24, MTU 9000")
        );
        write(sysfs, "class/net/enp1s0f0np0/mtu", "1500\n");
        let warning = inspect(true);
        assert!(warning.ok);
        assert_eq!(warning.checks[1].status, CheckStatus::Warn);
        assert!(
            warning.checks[1]
                .detail
                .contains("MTU 1500 is below 9000. MTU 9000 can improve throughput")
        );
        assert!(render(&warning, Canvas::sized(100, false)).contains("1 warning,"));
        fs::remove_file(port.join("state")).unwrap();
        assert!(!inspect(true).all_verified);
        assert_eq!(inspect(true).checks[1].status, CheckStatus::Unverified);
    }

    #[test]
    fn cx7_preflight_distinguishes_devices_beyond_the_serving_limit() {
        let dir = tempfile::tempdir().unwrap();
        let sysfs = dir.path();
        add_spark(sysfs);
        for i in 0..=MAX_DEVICES {
            let device = sysfs.join(format!("class/infiniband/roce{i}"));
            write(&device, "device/vendor", "0x15b3\n");
            write(&device, "device/device", "0x1021\n");
            write(&device, "ports/1/state", "4: ACTIVE\n");
            write(&device, "ports/1/gid_attrs/types/3", "RoCE v2\n");
            write(&device, "ports/1/gid_attrs/ndevs/3", &format!("eth{i}\n"));
            write(
                &device,
                "ports/1/gids/3",
                &format!("0000:0000:0000:0000:0000:ffff:c0a8:{i:02x}02\n"),
            );
            write(sysfs, &format!("class/net/eth{i}/mtu"), "9000\n");
        }
        let report = report(sysfs, |_, _| Some(24));
        let check = &report.checks[1];
        assert_eq!(check.status, CheckStatus::Pass, "{}", check.detail);
        assert!(check.detail.contains("4 of 5 active CX7 devices"));
        assert!(check.detail.contains("omitted roce4"));
    }

    #[test]
    fn cx7_subnets_accept_point_to_point_prefixes_and_flag_overlap() {
        let mut interfaces = [
            Interface {
                device: "roce0",
                interface: "enp1s0f0np0".into(),
                address: Ipv4Addr::new(192, 168, 0, 0),
                prefix: 31,
                mtu: Some(9000),
            },
            Interface {
                device: "roce1",
                interface: "enp1s0f1np1".into(),
                address: Ipv4Addr::new(192, 168, 1, 15),
                prefix: 24,
                mtu: Some(9000),
            },
        ];
        assert!(configuration_warnings(&interfaces).is_empty());
        interfaces[1].address = Ipv4Addr::new(192, 168, 0, 15);
        assert!(
            configuration_warnings(&interfaces)
                .join(" ")
                .contains("subnets overlap")
        );
        interfaces[1].prefix = 32;
        assert!(
            configuration_warnings(&interfaces)
                .join(" ")
                .contains("192.168.0.15/32")
        );
        interfaces[0].address = Ipv4Addr::new(169, 254, 1, 2);
        interfaces[1].address = Ipv4Addr::new(169, 254, 2, 2);
        interfaces.iter_mut().for_each(|port| port.prefix = 16);
        assert!(configuration_warnings(&interfaces).is_empty());
    }
}
