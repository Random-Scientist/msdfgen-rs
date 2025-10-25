use std::{array::from_fn, env};

use euclid::{
    Angle,
    approxord::{max, min},
    default::{Box2D, Transform2D},
};
use image::{ColorType, Pixel, Rgb, save_buffer, save_buffer_with_format};
use lyon_geom::{LineSegment, Point};
use msdfgen::{
    Contour, PathCollector, compute_msdf, compute_msdf_for, path::PathSegment, recolor_contours,
};
use ttf_parser::{Face, Rect};

fn get_font() -> Face<'static> {
    let bytes = include_bytes!("testfont.ttf");
    ttf_parser::Face::parse(bytes, 0).unwrap()
}

fn get_glyph(font: &Face<'static>, chr: char) -> (Vec<Contour>, Box2D<f32>) {
    let id = font.glyph_index(chr).unwrap();

    let mut builder = PathCollector::new();
    let _rect = font.outline_glyph(id, &mut builder).unwrap();
    let contours = builder.finish();
    let mut it = contours.iter().flat_map(|v| v.elements.iter());
    let init = it.next().map(|v| v.bounds()).unwrap_or(Box2D::zero());

    let rect = it.fold(init, |a, b| union(a, b.bounds()));
    // let rect = Box2D::new(
    //     Point::new(rect.x_min as f32, rect.y_min as f32),
    //     Point::new(rect.x_max as f32, rect.y_max as f32),
    // );
    (contours, rect)
}
fn union(a: Box2D<f32>, b: Box2D<f32>) -> Box2D<f32> {
    Box2D {
        min: Point::new(min(a.min.x, b.min.x), min(a.min.y, b.min.y)),
        max: Point::new(max(a.max.x, b.max.x), max(a.max.y, b.max.y)),
    }
}

pub fn main() {
    let Some(c) = std::env::args().nth(1).and_then(|v| v.chars().next()) else {
        return;
    };
    dbg!(c);
    // let font = get_font();
    // let (contour, rect) = get_glyph(&font, c);
    // let mut contour = recolor_contours(contour, Angle::degrees(3.0), 1);
    // let dim = 1000;

    // let scale = 1.0 / (rect.width().max(rect.height()));
    // dbg!(rect);
    // let transform = Transform2D::translation(-rect.min.x, -rect.min.y).then_scale(scale, scale);

    // contour
    //     .iter_mut()
    //     .for_each(|v| v.elements.iter_mut().for_each(|v| v.transform(&transform)));

    // let mut inner_rect = rect;
    // // transform the bounding rectangle down into (0,0) - (1,1)
    // inner_rect.min = transform.transform_point(inner_rect.min);
    // inner_rect.max = transform.transform_point(inner_rect.max);

    // let msdf = compute_msdf(&contour, dim as usize);

    // let buf = image::ImageBuffer::from_fn(dim, dim, |x, y| {
    //     let pix = msdf[x as usize + y as usize * dim as usize];
    //     //dbg!(pix);
    //     Rgb(pix.map(|v| ((v + 1.0) / 2.0 * 255.0) as u8))
    // });

    // let mut max_col = 0.0f32;
    // let mut min_col = 0.0f32;
    // let rendered = image::ImageBuffer::from_fn(dim, dim, |x, y| {
    //     let y = (dim - 1) - y;
    //     let pix = msdf[x as usize + y as usize * dim as usize];
    //     let col = pix[0].min(pix[1]).max(pix[0].max(pix[1]).min(pix[2]));
    //     max_col = max_col.max(col);
    //     min_col = min_col.min(col);
    //     let col = if col < 0.01 { 1.0 } else { 0.0 };
    //     Rgb(from_fn(|_| (col * 255.0) as u8))
    // });
    // dbg!(min_col, max_col);

    // rendered.save("rendered.png").unwrap();
    let s = make_glyph(c, 300);

    dbg!(&s.horizontal_lines, &s.vertical_lines);
    let rendered = image::ImageBuffer::from_fn(s.dims.0, s.dims.1, |x, y| {
        let y = (s.dims.1 - 1) - y;
        let pix = s.buf[x as usize + y as usize * s.dims.0 as usize];

        let col = pix[0].min(pix[1]).max(pix[0].max(pix[1]).min(pix[2]));
        let col = if col > 1.0 / 2000.0 { 1.0 } else { 0.0 };
        Rgb(from_fn(|_| (col * 255.0) as u8))
    });
    let buf = image::ImageBuffer::from_fn(s.dims.0, s.dims.1, |x, y| {
        let pix = s.buf[x as usize + y as usize * s.dims.0 as usize];
        //dbg!(pix);
        Rgb(pix.map(|v| ((v + 1.0) / 2.0 * 255.0) as u8))
    });
    rendered.save("rendered.png").unwrap();
    buf.save("out.png").unwrap();
}
#[derive(Debug)]
pub struct GlyphAndInfo {
    buf: Vec<[f32; 3]>,
    dims: (u32, u32),
    horizontal_lines: Vec<LineInfo>,
    vertical_lines: Vec<LineInfo>,
}
#[derive(Debug)]
pub struct LineInfo {
    axis_pos: f32,
    len: f32,
}
fn make_glyph(character: char, max_dim: u32) -> GlyphAndInfo {
    let (contour, glyph_bounds) = get_glyph(&get_font(), character);
    dbg!(glyph_bounds);
    let eps = 0.001;
    let (horiz, vert) = contour
        .iter()
        .flat_map(|v| v.elements.iter())
        .filter_map(|v| {
            if let PathSegment::Line(line) = v.segment
                && ((line.to - line.from)
                    .abs()
                    .lower_than(euclid::Vector2D::splat(eps)))
                .any()
            {
                Some(line)
            } else {
                None
            }
        })
        .partition::<Vec<_>, _>(|v| {
            let diff = v.to - v.from;
            diff.x < eps
        });
    let h = |vec: Vec<LineSegment<f32>>, horiz| -> Vec<LineInfo> {
        vec.into_iter()
            .map(|v| LineInfo {
                axis_pos: if horiz { v.from.x } else { v.from.y },
                len: v.length(),
            })
            .collect()
    };
    let horizontal_lines = h(horiz, true);
    let vertical_lines = h(vert, false);

    let mut contours = recolor_contours(contour, Angle::degrees(3.0), 1);
    let height = glyph_bounds.height();
    let width = glyph_bounds.width();
    let tall = height > width;

    let mut short_side = height / width;
    if tall {
        short_side = short_side.recip();
    }

    let max_size = if tall { height } else { width };
    let scale = 1.0 / max_size;
    dbg!(short_side, tall, scale);
    let v_size = if tall {
        max_dim
    } else {
        (max_dim as f32 * short_side).ceil() as u32
    };
    let h_size = if !tall {
        max_dim
    } else {
        (max_dim as f32 * short_side).ceil() as u32
    };
    dbg!(v_size, h_size);
    let scaled_in = scale * 0.90;

    let transform = Transform2D::translation(-glyph_bounds.min.x, -glyph_bounds.min.y)
        .then_scale(scaled_in, scaled_in)
        .then_translate(if tall {
            euclid::Vector2D::new(0.05 * short_side, 0.05)
        } else {
            euclid::Vector2D::new(0.05, 0.05 * short_side)
        });

    let outscale = 1.0 / max_dim as f32;
    contours.iter_mut().for_each(|v| v.transform(&transform));

    let buf = compute_msdf_for(
        &contours,
        (0..v_size).flat_map(|yoff| {
            let py = (yoff as f32 + 0.5) * outscale;
            (0..h_size).map(move |xoff| ((xoff as f32 + 0.5) * outscale, py))
        }),
    );
    dbg!(buf.len(), h_size * v_size);
    GlyphAndInfo {
        buf,
        dims: (h_size, v_size),
        horizontal_lines,
        vertical_lines,
    }
}
