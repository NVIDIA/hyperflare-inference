/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Static terminal startup banner: hand-crafted `hyperflare` wordmark with golden-amber 3D
//! wireframe shadow above a curved solar limb and diagonal solar flare eruption.

use std::fmt::Write as _;

pub(super) const WIDTH: usize = 80;
pub(super) const ROWS: usize = 16;

const ART: [&str; ROWS] = [
    "                                                           ⢀⢄·   ⣴⣿⣴⣴⣿⣫⣿⣿⣿⣿⣿⣿⣿⣿⣫",
    "         •                                         ✦       ⡼⠁  ⣠⣿⣷⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣝",
    " ██╗                           ███╗██╗                ·  ⢀⡼   ⣼⣿⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⣿⣟⡋⠉",
    " ██║                          ██╔═╝██║         ✧        ⢠⣾⠇⢸⣴⣾⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⡿⡛⠚⢉⣹⣾⠄ ",
    " ████╗ ██╗██╗████╗ ████╗ ████╗████╗██║████╗ ████╗████╗  ⣼⠏⡀⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠗⠊⠉    ",
    " ██╔██╗██║██║██╔██╗██▄██╗██╔═╝██╔═╝██║▄▄▄██╗██╔═╝██▄██╗⣼⣿⣠⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⡿⡟⠛⡛⠁       ",
    " ██║██║██║██║██║██║██╔══╝██║  ██║  ██║██╔██║██║  ██╔══╝⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣟⣋⡵⠿⠃ ·        ",
    " ██║██║╚████║████╔╝╚████╗██║  ██║  ██║╚████║██║  ╚████╗⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠉          ✧   ",
    " ╚═╝╚═╝▄▄▄██║██╔═╝  ╚═══╝╚═╝  ╚═╝  ╚═╝ ╚═══╝╚═╝   ╚═══╝⣿⣿⣿⣿⢿⠟⠉⠛⠻                ",
    "       ╚════╝╚═╝                 ·              ⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⠉⠁ ·           ✦      ",
    "                            ·      ⠒⠒ ⢤⣤⣄⡀  ⡀⢠⣤⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⡗⠒⢓⢂                   ",
    "                  ⣀⣀⣀⣠⣤⣤⣤⣴⣶⣶⣶⣶⣶⣿⠿⠿⢻⣿⣶⣿⣿⣬⣽⣻⣿⣿⣿⣾⣿⣿⣿⣿⣿⣿⣷⣿⣿⣿⣿⣭⣉⣉⣉⣉⡀   ·             ",
    "        ⣀⣀⣤⣤⣴⣶⣶⢻⢟⢟⡝⡢⢀⢖⡲⡄⡔⣢⢠⢖⢄⠰⡺⡍⡝⢫⢳⣞⣿⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⣿⣿⣟⡻⡍⡟⣻⢻⣷⣶⣶⣤⣤⣄⣀⡀        ",
    "⢀⣀⣤⣤⣶⣶⠿⠉⢓⣵⠈⠂⠵⠃⣓⠈⠂⠵⠃⠑⠥⣇⣑⠈⠂⠵⠃⠑⢥⣧⣑⡃⠳⠥⠇⢑⣡⣷⣕⠃⠽⡿⡟⢻⢿⡻⠛⢿⣿⡿⣿⣿⣿⠃⠮⠼⠝⣊⣼⣮⡚⠈⠈⠂⠵⠃⠑⢥⠣⢵⣇⣛⣉⣷⣶⣦⣤⣄⣀ ",
    "⡩⢤⠔⡒⢊⡭⣭⢚⡒⢿⠶⡀⠥⡞⢓⠺⡷⣿⠖⡋⢠⠭⡄⢉⡀⣿⢖⡀⠥⡞⢓⠺⡷⢌⠁⣨⢥⡜⣓⡀⠥⡾⢒⠚⣦⡀⠥⡞⢓⠺⡷⢌⠁⣨⢥⡜⣓⠺⡷⢎⠁⡨⢥⡉⣛⢵⡶⢢⠤⢄⡠⢤⠔⡒⢊⡭⣭⢚⡚⢿",
    "⣊⣸⠮⠞⠈⠪⠈⠂⠵⠃⠮⠼⠈⣊⣼⣮⡚⠈⠈⠂⠵⠃⠑⢥⠔ ⠣⢵⣇⣓⣁⠧⠵⠁⢓⣵⠈⠂⠵⠃⠵⠃⣓⠈⠂⠵⠃⠑⠥⣇⣑⠈⠂⠵⠃⠑⢥⣧⣑⡃⠳⠥⠇⢑⣡⣷⣕⠃⠱⠥⢇⣑⣊⣸⠮⠞⠈⠪⠈⠂",
];

const STYLES: [&str; ROWS] = [
    "                                                           223   336656888888876",
    "         R                                         S       43  55667888888887765",
    " WWG                           WWWGWWG                3  35   555788888888876534",
    " WWG                          WWGGGWWG         s        45436667888888876434533 ",
    " WWWWG WWGWWGWWWWG WWWWG WWWWGWWWWGWWGWWWWG WWWWGWWWWG  65346888888888755443    ",
    " WWGWWGWWGWWGWWGWWGWWWWWGWWGGGWWGGGWWGWWWWWGWWGGGWWWWWG664678888887765543       ",
    " WWGWWGWWGWWGWWGWWGWWGGGGWWG  WWG  WWGWWGWWGWWG  WWGGGG778888888754434 3        ",
    " WWGWWGGWWWWGWWWWGGGWWWWGWWG  WWG  WWGGWWWWGWWG  GWWWWG88888887644          s   ",
    " GGGGGGWWWWWGWWGGG  GGGGGGGG  GGG  GGG GGGGGGGG   GGGGG888865554                ",
    "       GGGGGGGGG                 3              578888888643 3           S      ",
    "                            2      43 4443  34568888887664333                   ",
    "                  333333334442454545666666777888888755555444342   2             ",
    "        3333333333333222222222222224555567888888888876676555454344443333        ",
    "3333333333444222221111114444444111113441688788778887755555411444444222333333333 ",
    "33222224455554411110000445555544000000000000000330453000000044555544144455444333",
    "11111114455554400000000040444440000000000000000000000000000044455444044555544112",
];

#[derive(Clone, Copy, PartialEq, Eq)]
struct Style {
    fg: Option<(u8, u8, u8)>,
    bold: bool,
}

fn palette(key: u8) -> Style {
    match key {
        b'W' => Style {
            fg: Some((248, 244, 228)),
            bold: true,
        },
        b'G' => Style {
            fg: Some((229, 162, 52)),
            bold: false,
        },
        b'R' => Style {
            fg: Some((235, 42, 30)),
            bold: true,
        },
        b'S' => Style {
            fg: Some((255, 226, 108)),
            bold: true,
        },
        b's' => Style {
            fg: Some((255, 198, 82)),
            bold: false,
        },
        b'0' => Style {
            fg: Some((132, 12, 10)),
            bold: false,
        },
        b'1' => Style {
            fg: Some((172, 20, 12)),
            bold: false,
        },
        b'2' => Style {
            fg: Some((214, 34, 14)),
            bold: false,
        },
        b'3' => Style {
            fg: Some((244, 68, 16)),
            bold: false,
        },
        b'4' => Style {
            fg: Some((255, 112, 22)),
            bold: false,
        },
        b'5' => Style {
            fg: Some((255, 158, 36)),
            bold: false,
        },
        b'6' => Style {
            fg: Some((255, 198, 56)),
            bold: false,
        },
        b'7' => Style {
            fg: Some((255, 238, 134)),
            bold: true,
        },
        b'8' => Style {
            fg: Some((255, 254, 240)),
            bold: true,
        },
        _ => Style {
            fg: None,
            bold: false,
        },
    }
}

pub(super) fn render(width: usize) -> String {
    let columns = width.min(WIDTH);
    let mut output = String::with_capacity(ROWS * (columns * 6 + 16));
    let default_style = Style {
        fg: None,
        bold: false,
    };
    for (art_line, style_line) in ART.iter().zip(STYLES.iter()) {
        // Trim trailing transparent sky cells so copy-paste and narrow terminals stay clean.
        let mut last_non_empty = 0;
        for (i, (ch, &key)) in art_line
            .chars()
            .zip(style_line.as_bytes().iter())
            .take(columns)
            .enumerate()
        {
            if ch != ' ' || key != b' ' {
                last_non_empty = i + 1;
            }
        }
        let mut active = default_style;
        for (character, &key) in art_line
            .chars()
            .zip(style_line.as_bytes().iter())
            .take(last_non_empty)
        {
            let next = palette(key);
            if next != active {
                match next.fg {
                    Some((r, g, b)) => {
                        let weight = if next.bold { 1 } else { 22 };
                        let _ = write!(output, "\x1b[{weight};38;2;{r};{g};{b}m");
                    }
                    None => output.push_str("\x1b[0m"),
                }
                active = next;
            }
            output.push(character);
        }
        if active != default_style {
            output.push_str("\x1b[0m");
        }
        output.push('\n');
    }
    output.push('\n');
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip_ansi(text: &str) -> String {
        let mut out = String::new();
        let mut chars = text.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                for c in chars.by_ref() {
                    if c == 'm' {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn art_and_style_tables_have_matching_dimensions_and_valid_keys() {
        for (row, (art, styles)) in ART.iter().zip(STYLES.iter()).enumerate() {
            assert_eq!(art.chars().count(), WIDTH, "art row {row}");
            assert_eq!(styles.len(), WIDTH, "styles row {row}");
            for &b in styles.as_bytes() {
                assert!(
                    matches!(b, b' ' | b'W' | b'G' | b'R' | b'S' | b's' | b'0'..=b'8'),
                    "unexpected style key {:?} on row {row}",
                    b as char
                );
            }
        }
    }

    #[test]
    fn static_banner_respects_terminal_width_and_resets_styles() {
        for width in [0, 1, 40, WIDTH, 120] {
            let rendered = render(width);
            assert_eq!(rendered.lines().count(), ROWS + 1, "width={width}");
            for line in rendered.lines() {
                let visible = strip_ansi(line);
                assert!(
                    visible.chars().count() <= width.min(WIDTH),
                    "width={width}: {visible:?}"
                );
                if line.contains("\x1b[") {
                    assert!(line.ends_with("\x1b[0m"), "unterminated SGR: {line:?}");
                }
            }
        }
    }
}
