use cgmath::{Point3, Vector3, SquareMatrix, Transform, InnerSpace};
use crate::camera::{Camera, Projection};
use crate::collision::{Collider, BVTree};
use crate::aab::{AxisAlignedBounding};

const PRECISION: f32 = 0.5;
const MULTIPLICITY: u32 = 1000;

pub struct Ray {
    origin: Point3<f32>,
    direction: Vector3<f32>,
    precision: f32,
    multiplicity: u32,
}


impl Ray {
    pub fn screen_coord_to_ray(camera: &Camera, projection: &Projection, 
            config: &wgpu::SurfaceConfiguration, mouse_x: f64, mouse_y: f64) -> Self {

        let mouse_coord = cgmath::point2(mouse_x, mouse_y);

        let ndc = cgmath::vec2(
            2.0 * mouse_coord.x as f32 / config.width as f32 - 1.0,
            1.0 - 2.0 * mouse_coord.y as f32 / config.height as f32,
        );

        let inv_proj_matrix = projection.calc_matrix().invert().unwrap();

        let clip_pos = Point3::new(ndc.x, ndc.y, -1.0);
        let clip_pos = inv_proj_matrix.transform_point(clip_pos);
        let clip_pos = Vector3::new(clip_pos.x, clip_pos.y, clip_pos.z);

        let inv_view_matrix = camera.calc_matrix().invert().unwrap();

        let direction = inv_view_matrix.transform_vector(clip_pos).normalize();

        // Create a ray in world space using the camera position and ray direction.
        let origin = camera.position;

        Ray {
            origin,
            direction,
            precision: PRECISION,
            multiplicity: MULTIPLICITY,
        }
    }

    pub fn find_closest_object<'a, T, V>(&'a self, tree: &'a BVTree<T, V>) -> Option<V>
        where T: AxisAlignedBounding<V, Aab = T> + std::fmt::Debug + Clone, V : Collider + Clone {
        let mut ray_current_pos = self.origin.clone();
        let mut intersected_object = None;
        for _ in 0..self.multiplicity {
            let near = tree.point_intersection(&ray_current_pos);
            match near {
                Some(..) => {
                    intersected_object = near;
                    break;
                },
                None => {
                    ray_current_pos += self.direction * self.precision;
                }
            }
        }

        return intersected_object;
    }
}