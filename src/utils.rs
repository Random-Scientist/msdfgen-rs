//! Utilities for the SDF computations

use lyon_geom::Vector;

use crate::path::PathElement;
/// Represents a distance to an edge segment
#[derive(Copy, Clone)]
pub(crate) struct EdgeDistance<'a> {
    pub(crate) dist: AugmentedDistance,
    pub(crate) edge: Option<&'a PathElement>,
    pub(crate) nearest_approach: f32,
}

impl<'a> EdgeDistance<'a> {
    /// Create a new edge point, initialized to infinite distance
    pub(crate) fn new() -> Self {
        Self {
            dist: AugmentedDistance::new(-1e24, 0.0),
            edge: None,
            nearest_approach: 0.0,
        }
    }

    pub(crate) fn mk_pseudodistance(&mut self, p: Vector<f32>) {
        if let Some(edge) = self.edge {
            self.dist = edge.mk_pseudodistance(self.dist, p, self.nearest_approach)
        }
    }
}

/// A signed distance, augmented with the cosine of the angle
/// between the tangent of the edge and the vector from the
/// point of nearest approach to the measured point.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct AugmentedDistance {
    /// The actual distance
    pub(crate) distance: f32,
    /// The cosine of the angle between the tangent vector of the path segment
    /// at the point of closest approach and the vector from the point of
    /// closest approach to the point to which distance was measured. This is used to
    dot: f32,
}

impl AugmentedDistance {
    pub(crate) fn new(distance: f32, dot: f32) -> Self {
        Self { distance, dot }
    }
}

impl std::cmp::PartialOrd for AugmentedDistance {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        match self.distance.abs().partial_cmp(&other.distance.abs()) {
            Some(Ordering::Less) => Some(Ordering::Less),
            Some(Ordering::Greater) => Some(Ordering::Greater),
            Some(Ordering::Equal) => self.dot.partial_cmp(&other.dot),
            None => None,
        }
    }
}
