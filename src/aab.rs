use cgmath::{Point3, Vector3};
use crate::collision::{Collider};
use crate::ray::{Ray};
use std::f32::consts::PI;

pub(crate) const PADDING:f32 = 0.25;

pub trait AxisAlignedBounding<V: Collider> {
    type Aab: AxisAlignedBounding<V>;
    fn position(&self) -> Point3<f32>;
    fn padding(&self) -> f32;
    fn create_aab(collider: &V) -> Self::Aab;
    fn create_aab_with_padding(collider: &V, padding: f32) -> Self::Aab;
    fn intersect(&self, other: &Self::Aab) -> bool;
    fn point_intersects(&self, point: &Point3<f32>) -> bool;
    fn intersection_point(&self, ray: &Ray, point: &Point3<f32>) -> Option<Point3<f32>>;
    fn merge(&self, other: &Self::Aab) -> Self;
    fn surface_area_cost(&self) -> f32;
}
#[derive(Debug)]
pub struct AABSphere {
    center: Point3<f32>,
    radius: f32,
    squared_radius: f32,
}
impl <V: Collider> AxisAlignedBounding<V> for AABSphere {
    type Aab = AABSphere;
    fn position(&self) -> Point3<f32> {
        self.center
    }

    fn padding(&self) -> f32 {
        todo!();
    }

    fn create_aab(collider: &V) -> Self::Aab {
        let radius = 1.0 * collider.scale().x.max(collider.scale().y).max(collider.scale().z);
        AABSphere {
            center: collider.position(),
            radius,
            squared_radius: radius.powf(2.0),
        }
    }

    fn create_aab_with_padding(collider: &V, padding: f32) -> Self::Aab {
        todo!();
    }
    
    fn intersect(&self, other: &AABSphere) -> bool {
        // Squared Euclian of point - center ^ 2 compared to squared radius
        let adjust = other.center - self.center;
        adjust.x * adjust.x + adjust.y * adjust.y + adjust.z * adjust.z <= self.squared_radius
    }

    fn point_intersects(&self, point: &Point3<f32>) -> bool {
        todo!();
    }

    fn intersection_point(&self, _ray: &Ray, _point: &Point3<f32>) -> Option<Point3<f32>> {
        todo!();
    }

    fn merge(&self, other: &AABSphere) -> Self {
        let added_radius = self.radius + other.radius;
        let self_center = self.center * self.radius;
        let other_center = other.center * other.radius;
        let added_centers = Point3 { 
            x: (self_center.x + other_center.x) / 2.0,
            y: (self_center.y + other_center.y) / 2.0,
            z: (self_center.z + other_center.z) / 2.0,
        };
        Self {
            center: added_centers / added_radius,
            radius: added_radius / 2.0,
            squared_radius: (added_radius / 2.0).powf(2.0),
        }
    }

    fn surface_area_cost(&self) -> f32 {
        // This is just the surface area of a sphere
        4.0 * PI * self.squared_radius
    }
}

#[derive(Clone)]
pub struct AABRect {
    center: Point3<f32>,
    min_corner: Point3<f32>,
    max_corner: Point3<f32>,
    scale: Vector3<f32>,
    padding: f32,
}
impl AABRect {
    pub fn to_raw(&self) -> WireframeRaw {
        let model =
            cgmath::Matrix4::from_translation(cgmath::Vector3 {
                x: self.center.x,
                y: self.center.y,
                z: self.center.z
            });
        WireframeRaw {
            model: model.into(),
            scale: self.scale.into(),
        }
    }
}

impl <V: Collider> AxisAlignedBounding<V> for AABRect {
    type Aab = AABRect;
    fn position(&self) -> Point3<f32> {
        self.center
    }

    fn padding(&self) -> f32 {
        self.padding
    }

    fn create_aab(collider: &V) -> Self::Aab {
        Self::create_aab_with_padding(collider, PADDING)
    }

    fn create_aab_with_padding(collider: &V, padding: f32) -> Self::Aab {
        let scale = Vector3 { x: collider.scale().x + PADDING, y: collider.scale().y + PADDING, z: collider.scale().z + PADDING };
        Self {
            center: collider.position(),
            min_corner: Point3 { x: collider.position().x - scale.x, y: collider.position().y - scale.y, z: collider.position().z - scale.z }, 
            max_corner: Point3 { x: collider.position().x + scale.x, y: collider.position().y + scale.y, z: collider.position().z + scale.z },
            scale,
            padding,
        }
    }
    
    fn intersect(&self, other: &AABRect) -> bool {
        self.min_corner.x < other.max_corner.x && self.max_corner.x > other.min_corner.x 
            && self.min_corner.y < other.max_corner.y && self.max_corner.y > other.min_corner.y 
                && self.min_corner.z < other.max_corner.z && self.max_corner.z > other.min_corner.z
    }

    fn point_intersects(&self, point: &Point3<f32>) -> bool {
        self.min_corner.x < point.x && self.max_corner.x > point.x
            && self.min_corner.y < point.y && self.max_corner.y > point.y
                && self.min_corner.z < point.z && self.max_corner.z > point.z
    }

    fn intersection_point(&self, _ray: &Ray, _point: &Point3<f32>) -> Option<Point3<f32>> {
        todo!();
    }

    fn merge(&self, other: &Self::Aab) -> Self {
        let min_extent = Point3 {
            x: self.min_corner.x.min(other.min_corner.x),
            y: self.min_corner.y.min(other.min_corner.y),
            z: self.min_corner.z.min(other.min_corner.z),
        };
        let max_extent = Point3 {
            x: self.max_corner.x.max(other.max_corner.x),
            y: self.max_corner.y.max(other.max_corner.y),
            z: self.max_corner.z.max(other.max_corner.z),
        };
        let center = Point3 {
            x: ((max_extent.x - min_extent.x) / 2.0) + min_extent.x,
            y: ((max_extent.y - min_extent.y) / 2.0) + min_extent.y,
            z: ((max_extent.z - min_extent.z) / 2.0) + min_extent.z,
        };
        let scale = Vector3 {
            x: ((max_extent.x - min_extent.x) / 2.0) + self.padding,
            y: ((max_extent.y - min_extent.y) / 2.0) + self.padding,
            z: ((max_extent.z - min_extent.z) / 2.0) + self.padding,
        };
        Self {
            center,
            min_corner: min_extent,
            max_corner: max_extent,
            scale,
            padding: self.padding
        }
    }

    fn surface_area_cost(&self) -> f32 {
        let width = self.max_corner.x - self.min_corner.x;
        let height = self.max_corner.y - self.min_corner.y;
        let depth = self.max_corner.z - self.min_corner.z;
        width * height * depth
    }
}
impl std::fmt::Debug for AABRect {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        print!(
            "center: {:?}, min: {:?}, max: {:?}, scale: {:?}", 
            vec![self.center.x, self.center.y, self.center.z],
            vec![self.min_corner.x, self.min_corner.y, self.min_corner.z],
            vec![self.max_corner.x, self.max_corner.y, self.max_corner.z],
            vec![self.scale.x, self.scale.y, self.scale.z],
        );
        Ok(())
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WireframeRaw {
    pub model: [[f32; 4]; 4],
    pub scale: [f32; 3],
}

impl WireframeRaw {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<WireframeRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    // While our vertex shader only uses locations 0, and 1 now, in later tutorials we'll
                    // be using 2, 3, and 4, for Vertex. We'll start at slot 5 not conflict with them later
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                }
            ],
        }
    }
}