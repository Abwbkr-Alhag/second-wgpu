use cgmath::{Point3, Vector3, InnerSpace};
use std::{f32::NEG_INFINITY, io::{Cursor, BufReader, BufRead}};
use crate::resources::load_string;

const O: Vector3<f32> = Vector3{ x: 0.0, y: 0.0, z: 0.0 };

pub struct GJKModel {
    pub shapes: Vec<Box<dyn GJKCollider>>
}

use std::fmt;

impl fmt::Debug for GJKModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print each shape using the Debug trait
        for shape in &self.shapes {
            write!(f, "{:?}\n", shape)?;
        }
        Ok(())
    }
}

impl fmt::Debug for dyn GJKCollider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GJKCollider: {:?}", self.debug_info())
    }
}

pub trait GJKCollider {
    fn debug_info(&self) -> String;
    fn find_furthest_point(&self, direction: &Vector3<f32>) -> Vector3<f32>;
    fn get_moved_and_scaled_position(&self, offset: Vector3<f32>, scale: Vector3<f32>) -> Box<dyn GJKCollider>;
    fn contains_point(&self, point: &Point3<f32>) -> bool;
}

pub struct Simplex {
    pub points: [Vector3<f32>; 4],
    size: usize,
}

impl Simplex {
    fn new() -> Simplex {
        Simplex {
            points: [Vector3 { x: 0.0, y: 0.0, z: 0.0 };4],
            size: 0,
        }
    }

    pub fn add(&mut self, new_point: Vector3<f32>) {
        self.points = [new_point, self.points[0], self.points[1], self.points[2]];
        self.size = 4.min(self.size + 1);
    }

    pub fn set(&mut self, new_points: Vec<Vector3<f32>>) {
        if new_points.len() > 4 {
            panic!("Trying to set a simplex with more than 4 points")
        }
        for (i, point) in new_points.iter().enumerate() {
            self.points[i] = *point;
        }
        self.size = new_points.len();
    }
}

// All of these structs are using points with respect to the world coordinates
// However how these are constructed will be based off the objects coordinates itself
#[derive(Copy, Clone)]
struct Sphere {
    position: Point3<f32>,
    radius: f32,
}

impl GJKCollider for Sphere {
    fn debug_info(&self) -> String {
        format!("Center: {:?}, Radius: {}", self.position, self.radius)
    }
    fn find_furthest_point(&self, direction: &Vector3<f32>) -> Vector3<f32> {
        let point = self.position + (direction.normalize() * self.radius);
        Vector3 { x: point.x, y: point.y, z: point.z }
    }
    fn get_moved_and_scaled_position(&self, offset: Vector3<f32>, scale: Vector3<f32>) -> Box<dyn GJKCollider> {
        if scale.x != scale.y && scale.y != scale.z {
            panic!("Sphere has to be scaled equally!");
        }
        Box::new(Self {
            position: self.position + offset,
            radius: self.radius * scale.x,
        })
    }
    fn contains_point(&self, point: &Point3<f32>) -> bool {
        let vector = self.position - point;
        let magnitude = vector.magnitude();
        self.radius <= magnitude
    }
}

#[derive(Clone, Debug)]
struct Face {
    vertices: Vec<Vector3<f32>>,
    normal: Vector3<f32>,
}

#[derive(Clone)]
struct Polygon {
    faces: Vec<Face>,
}


impl GJKCollider for Polygon {
    fn debug_info(&self) -> String {
        format!("{:?}", self.faces)
    }
    fn find_furthest_point(&self, direction: &Vector3<f32>) -> Vector3<f32> {
        let mut max_vertex = Vector3 { x: 0.0, y: 0.0, z: 0.0 };
        let mut max_dist = NEG_INFINITY;
        for face in self.faces.as_slice() {
            for vertex in face.vertices.as_slice() {
                let distance = vertex.dot(*direction);
                if distance > max_dist {
                    max_dist = distance;
                    max_vertex = *vertex
                }
            }
        }
		max_vertex
    }
    fn get_moved_and_scaled_position(&self, offset: Vector3<f32>, scale: Vector3<f32>) -> Box<dyn GJKCollider> {
        let mut faces = vec![];
        for face in self.faces.iter() {
            let mut vertices = vec![];
            for vertex in face.vertices.iter() {
                let scaled_vertex = Vector3 { x: vertex.x * scale.x, y: vertex.y * scale.y, z: vertex.z * scale.z };
                vertices.push(scaled_vertex + offset);
            }
            // calc normals again
            // WARNING: ASSUMES EACH POLYGON'S FACE HAS 3 VERTICES
            let a = vertices[0];
            let b = vertices[1];
            let c = vertices[2];
            let ab = b - a;
            let ac = c - a;
            let mut normal = ab.cross(ac).normalize();
            if normal.dot(face.normal) < 0.0 {
                normal = -normal;
            }
            faces.push(Face { vertices, normal });
        }
        Box::new(Self {
            faces,
        })
    }
    fn contains_point(&self, point: &Point3<f32>) -> bool {
        let point: Vector3<f32> = Vector3 { x: point.x, y: point.y, z: point.z };
        // ASSUME EACH FACE HAS A VERTICE
        for face in self.faces.iter() {
            let vertex = face.vertices[0];
            let direction: Vector3<f32> = point - vertex;
            //  point lines inside an object than it will be in an opposite direction of the normal
            if direction.dot(face.normal) > 0.0 {
                return false;
            }
        }
        true
    }
}

pub fn support<G: GJKCollider + ?Sized>(collider_a: &G, collider_b: &G, direction: &Vector3<f32>) -> Vector3<f32> {
    let opposite = Vector3 { x: -direction.x, y: -direction.y, z: -direction.z };
    collider_a.find_furthest_point(direction) - collider_b.find_furthest_point(&opposite)
}

pub fn gjk<G: GJKCollider + ?Sized>(collider_a: &G, collider_b: &G) -> Option<Simplex> {
    let initial_vec = support(collider_a, collider_b, &Vector3::unit_x());
    let mut simplex = Simplex::new();
    simplex.add(initial_vec);

    let mut direction = -initial_vec;

    loop {
        let support = support(collider_a, collider_b, &mut direction);

        if support.dot(direction) <= 0.0 {
            return None;
        }
        simplex.add(support);

        if next_simplex(&mut simplex, &mut direction) {
            return Some(simplex);
        }
    }

}

fn next_simplex(simplex: &mut Simplex, direction: &mut Vector3<f32>) -> bool {
    match simplex.size {
        2 => return line_case       (simplex, direction),
        3 => return triangle_case   (simplex, direction),
        4 => return tetrahedron_case(simplex, direction),
        _ => panic!("Can't have a simplex of this size!"),
    }
}

fn line_case(simplex: &mut Simplex, direction: &mut Vector3<f32>) -> bool {
    let a = simplex.points[0];
    let b = simplex.points[1];

    let ab = b - a;
    let ao = O - a;

    if ab.dot(ao) >= 0.0 {
        *direction = ab.cross(ao).cross(ab);
    } else {
        simplex.set(vec![a]);
        *direction = ao;
    }

    false
}

fn triangle_case(simplex: &mut Simplex, direction: &mut Vector3<f32>) -> bool {
    let a = simplex.points[0];
    let b = simplex.points[1];
    let c = simplex.points[2];

    let ab = b - a;
    let ac = c - a;
    let ao = O - a;

    let abc = ab.cross(ac);

    if (abc.cross(ac)).dot(ao) >= 0.0 {
        if ac.dot(ao) >= 0.0 {
            simplex.set(vec![a, c]);
            *direction = ac.cross(ao).cross(ac);
        } else {
            simplex.set(vec![a, b]);
            return line_case(simplex, direction);
        }
    } else {
        if (ab.cross(abc)).dot(ao) >= 0.0 {
            simplex.set(vec![a, b]);
            return line_case(simplex, direction);
        } else {
            if abc.dot(ao) >= 0.0 {
                *direction = abc;
            } else {
                simplex.set(vec![a, c, b]);
                *direction = -abc;
            }
        }
    }

    false
}

fn tetrahedron_case(simplex: &mut Simplex, direction: &mut Vector3<f32>) -> bool {
    let a = simplex.points[0];
    let b = simplex.points[1];
    let c = simplex.points[2];
    let d = simplex.points[3];

    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    let ao = O - a;
    
    let abc = ab.cross(ac);
    let acd = ac.cross(ad);
    let adb = ad.cross(ab);

    if abc.dot(ao) >= 0.0 {
        simplex.set(vec![a, b, c]);
		return triangle_case(simplex, direction);
	}
	if acd.dot(ao) >= 0.0 {
        simplex.set(vec![a, c, d]);
		return triangle_case(simplex, direction);
	}
 
	if adb.dot(ao) >= 0.0 {
        simplex.set(vec![a, d, b]);
		return triangle_case(simplex, direction);
	}
 
	true
}

fn parse_vertices(line: &str, arr: &mut Vec<Vector3<f32>>) {
    let mut vertex = [0.0;3];
    let split = line.split(" ");
    for (i, float) in split.enumerate() {
        if i > 2 {
            panic!("Out of bounds!");
        }
        vertex[i] = float.parse().expect("Cannot parse this line!");
    }
    arr.push(vertex.into());
}

fn parse_face(line: &str, polygon: &mut Polygon, normals: &Vec<Vector3<f32>>, vertices: &Vec<Vector3<f32>>) {
    let mut face = Face { vertices: vec![], normal: Vector3 { x: 0.0, y: 0.0, z: 0.0 } };
    let split = line.split("/");
    let mut vectors: Vec<Vector3<f32>> = Vec::new();
    for (_, float) in split.enumerate() {
        if float.split(" ").collect::<Vec<&str>>().len() > 1 {
            let last:Vec<&str> = float.split(" ").collect();
            let index: usize = last[0].parse::<usize>().expect("Cannot parse this line!") - 1;
            vectors.push(vertices[index]);
            let index: usize = last[1].parse::<usize>().expect("Can't parse!") - 1;
            face.normal = normals[index];
            break;
        }
        let index: usize = float.parse::<usize>().expect("Cannot parse this line!") - 1;
        vectors.push(vertices[index]);
    }
    face.vertices = vectors;
    polygon.faces.push(face);
}

fn parse_circle(line: &str, gjk_model: &mut GJKModel) {
    let mut vertex = [0.0;4];
    let split = line.split(" ");
    let mut sphere = Sphere { position: Point3 { x: 0.0, y: 0.0, z: 0.0 }, radius: 0.0 };
    for (i, float) in split.enumerate() {
        if i > 3 {
            panic!("Out of bounds!");
        }
        vertex[i] = float.parse().expect("Cannot parse this line!");
    }
    sphere.position = Point3 { x: vertex[0], y: vertex[1], z: vertex[2] };
    sphere.radius = vertex[3];
    gjk_model.shapes.push(Box::new(sphere));
}


pub async fn load_gjk_model(
    file_name: &str
) -> GJKModel {
    let obj_text = load_string(file_name).await.expect("Cannot load file");
    let obj_cursor = Cursor::new(obj_text);
    let obj_reader = BufReader::new(obj_cursor);
    let mut gjk_model: GJKModel = GJKModel { shapes: vec![] };
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut polygon = Polygon { faces: vec![] };

    for line in obj_reader.lines() {
        let (line, mut words) = match line {
            Ok(ref line) => (&line[..], line[..].split_whitespace()),
            Err(e) => {
                panic!("{}", e);
            }
        };
        match words.next() {
            Some("v") => {
                let rest = line[1..].trim();
                parse_vertices(rest, &mut vertices);
            },
            Some("vn") => {
                let rest = line[2..].trim();
                parse_vertices(rest, &mut normals);
            },
            Some("poly") => {
                // That's the end of that polgyon
                gjk_model.shapes.push(Box::new(polygon.clone()));
                polygon = Polygon { faces: vec![] };
            },
            Some("f") => {
                let rest = line[1..].trim();
                parse_face(rest, &mut polygon, &normals, &vertices);
            },
            Some("c") => {
                // We're going to encounter a sphere
                let rest = line[1..].trim();
                parse_circle(rest, &mut gjk_model);
            }
            Some(&_) => continue,
            None => continue
        }
    }
    gjk_model
}
