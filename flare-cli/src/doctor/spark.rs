/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use std::{fs, io, net::Ipv4Addr, path::Path};

use flare_helios::execution::roce_fabric::Fabric;

use super::{Check, CheckStatus};

// PCI IDs in NVML format (device in the upper half, vendor in the lower half).
const DGX_SPARK_GB10: u32 = 0x2e12_10de;
const CONNECTX_7: u32 = 0x1021_15b3;
const JUMBO_MTU: u32 = 9000;

pub(super) fn is_gb10(sysfs: &Path) -> bool {
    fs::read_dir(sysfs.join("bus/pci/devices")).is_ok_and(|devices| {
        devices
            .filter_map(Result::ok)
            .any(|device| pci_id(&device.path()) == Some(DGX_SPARK_GB10))
    })
}

fn pci_id(device: &Path) -> Option<u32> {
    let read_id =
        |name| u16::from_str_radix(read(&device.join(name))?.strip_prefix("0x")?, 16).ok();
    Some((u32::from(read_id("device")?) << 16) | u32::from(read_id("vendor")?))
}

fn read(file: &Path) -> Option<String> {
    fs::read_to_string(file).ok().map(|s| s.trim().to_owned())
}

pub(super) fn check_fabric(
    sysfs: &Path,
    discover: impl FnOnce() -> flare_helios::Result<Fabric>,
) -> Check {
    let (status, detail, remediation) = match inspect(sysfs, discover) {
        Ok(outcome) => outcome,
        Err(error) => (
            CheckStatus::Unverified,
            error.to_string(),
            Some("Make the host's RDMA and network sysfs attributes visible to flare doctor."),
        ),
    };
    Check {
        id: "spark.cx7".into(),
        name: "DGX Spark ConnectX-7 fabric".into(),
        status,
        expected: Some("An active CX7 link with an assigned IPv4 address and a RoCEv2 GID".into()),
        actual: None,
        detail,
        remediation: remediation.map(str::to_owned),
    }
}

fn inspect(
    sysfs: &Path,
    discover: impl FnOnce() -> flare_helios::Result<Fabric>,
) -> io::Result<(CheckStatus, String, Option<&'static str>)> {
    let root = sysfs.join("class/infiniband");
    let mut devices = match fs::read_dir(&root) {
        Ok(entries) => entries.collect::<io::Result<Vec<_>>>()?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error),
    };
    devices.retain(|device| pci_id(&device.path().join("device")) == Some(CONNECTX_7));
    devices.sort_by_key(|device| device.file_name());
    if devices.is_empty() {
        return Ok((
            CheckStatus::Fail,
            "No ConnectX-7 RDMA devices found.".into(),
            Some(
                "Check that the ConnectX-7 adapter is visible and the mlx5 RDMA driver is loaded.",
            ),
        ));
    }

    let mut active = Vec::new();
    let mut unreadable = Vec::new();
    for device in &devices {
        let name = device.file_name().to_string_lossy().into_owned();
        match read(&device.path().join("ports/1/state")) {
            Some(state)
                if state
                    .split_once(':')
                    .is_some_and(|(_, s)| s.trim() == "ACTIVE") =>
            {
                active.push(name);
            }
            Some(_) => {}
            None => unreadable.push(name),
        }
    }
    if active.is_empty() {
        if !unreadable.is_empty() {
            return Ok((
                CheckStatus::Unverified,
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
        return Ok((
            CheckStatus::Fail,
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
            let interface = read(&netdev).unwrap_or_else(|| "interface unavailable".into());
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
                "{} of {} active CX7 devices have an assigned IPv4 address at RoCEv2 GID index {}: {listed}.",
                ready.len(),
                active.len(),
                fabric.gid_index
            );
            let mut warnings = configuration_warnings(&ready);
            if ready.len() < active.len() {
                let missing: Vec<_> = active
                    .iter()
                    .filter(|name| !ready.iter().any(|port| port.device == name.as_str()))
                    .map(String::as_str)
                    .collect();
                warnings.push(format!(
                    "Check the IPv4 address and GID on {}.",
                    missing.join(", ")
                ));
            }
            if warnings.is_empty() {
                return Ok((CheckStatus::Pass, detail, None));
            }
            detail.push(' ');
            detail.push_str(&warnings.join(" "));
            return Ok((
                CheckStatus::Warn,
                detail,
                Some(
                    "Use IPv4 subnets shared with the intended peers. Confirm jumbo MTU support on the peers and switch.",
                ),
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
    Ok((
        CheckStatus::Fail,
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

fn subnet_mask(prefix: u8) -> u32 {
    u32::MAX.checked_shl(32 - u32::from(prefix)).unwrap_or(0)
}

fn configuration_warnings(ports: &[Interface<'_>]) -> Vec<String> {
    let mut warnings = Vec::new();
    for (i, port) in ports.iter().enumerate() {
        let name = &port.interface;
        let address = u32::from(port.address);
        let host_bits = !subnet_mask(port.prefix);
        let host = address & host_bits;
        if matches!(port.prefix, 0 | 32)
            || port.address.is_unspecified()
            || port.address.is_loopback()
            || port.address.is_multicast()
            || port.address.is_broadcast()
            || (port.prefix < 31 && (host == 0 || host == host_bits))
        {
            warnings.push(format!(
                "Review {name}'s IPv4 subnet ({}/{}).",
                port.address, port.prefix
            ));
        } else if port.address.is_link_local() {
            warnings.push(format!(
                "{name} uses link-local IPv4, which requires an LLDP-confirmed direct peer."
            ));
        }

        match port.mtu {
            Some(mtu) if mtu < JUMBO_MTU => {
                warnings.push(format!(
                    "{name}'s MTU {mtu} is below the recommended {JUMBO_MTU}."
                ));
            }
            None => warnings.push(format!("Could not read the MTU for {name}.")),
            _ => {}
        }

        for other in &ports[..i] {
            if port.interface == other.interface {
                continue;
            }
            let mask = subnet_mask(port.prefix.min(other.prefix));
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

#[cfg(test)]
mod tests {
    use super::*;
    use flare_helios::execution::roce_fabric;

    fn write(root: &Path, path: &str, content: &str) {
        let file = root.join(path);
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(file, content).unwrap();
    }

    #[test]
    fn cx7_readiness_requires_a_live_ipv4_address_at_the_serving_gid_index() {
        let dir = tempfile::tempdir().unwrap();
        let sysfs = dir.path();
        let root = sysfs.join("class/infiniband");
        let port = root.join("roce0/ports/1");
        write(&root, "roce0/device/vendor", "0x15b3\n");
        write(&root, "roce0/device/device", "0x1021\n");
        write(&root, "roce1/device/vendor", "0x15b3\n");
        write(&root, "roce1/device/device", "0x1021\n");
        write(&root, "roce1/ports/1/state", "1: DOWN\n");
        write(&port, "state", "4: ACTIVE\n");
        write(&port, "gid_attrs/ndevs/3", "enp1s0f0np0\n");
        write(&port, "gids/3", "0000:0000:0000:0000:0000:ffff:c0a8:0b02\n");
        write(sysfs, "class/net/enp1s0f0np0/mtu", "9000\n");

        let inspect = |has_ip| {
            check_fabric(sysfs, || {
                roce_fabric::discover_in(
                    &root,
                    |netdev, addr| {
                        (has_ip && netdev == "enp1s0f0np0" && addr.octets() == [192, 168, 11, 2])
                            .then_some(24)
                    },
                    |_| Default::default(),
                )
            })
        };
        write(&port, "gid_attrs/types/3", "RoCE v1\n");
        assert_eq!(inspect(true).status, CheckStatus::Fail);
        write(&port, "gid_attrs/types/3", "RoCE v2\n");
        assert_eq!(inspect(false).status, CheckStatus::Fail);
        let ready = inspect(true);
        assert_eq!(ready.status, CheckStatus::Pass, "{}", ready.detail);
        assert!(
            ready
                .detail
                .contains("enp1s0f0np0, 192.168.11.2/24, MTU 9000")
        );
        write(sysfs, "class/net/enp1s0f0np0/mtu", "1500\n");
        let warning = inspect(true);
        assert_eq!(warning.status, CheckStatus::Warn);
        assert!(
            warning
                .detail
                .contains("MTU 1500 is below the recommended 9000")
        );
        fs::remove_file(port.join("state")).unwrap();
        assert_eq!(inspect(true).status, CheckStatus::Unverified);
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
    }
}
