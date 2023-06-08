use crate::gjk::{Simplex, GJKCollider, support};
use cgmath::{Vector3, InnerSpace};

const EPA_MAX_ITER: usize = 10;

pub struct Polytope {
    points: Vec<Vector3<f32>>,
}

#[derive(Clone, Debug)]
pub struct CollisionVector {
	pub normal: Vector3<f32>,
	pub depth: f32,
}

impl Polytope {
    pub fn new(simplex: &Simplex) -> Polytope  {
        let points = vec![simplex.points[0], simplex.points[1], simplex.points[2], simplex.points[3]];
        Polytope {
            points
        }
    }
}

pub fn epa<G: GJKCollider + ?Sized>(collider_a: &G, collider_b: &G, simplex: &Simplex) -> Option<CollisionVector> {
    let mut polytope = Polytope::new(simplex);
    let mut faces = vec![
        0, 1, 2,
        0, 3, 1,
        0, 2, 3,
        1, 3, 2
    ];

    let (mut normals, mut min_face) = get_face_normals(&polytope, &faces);
    let mut min_normal = Vector3 { x: 0.0, y: 0.0, z: 0.0 };
    let mut min_distance = f32::INFINITY;

    let mut cnt = 0;
    while min_distance == f32::INFINITY {
        min_normal = normals[min_face as usize].0;
        min_distance = normals[min_face as usize].1;

        if cnt > EPA_MAX_ITER {
            break;
        }
        cnt+=1;

        let support = support(collider_a, collider_b, &min_normal);
        let s_distance = min_normal.dot(support);

        if (s_distance - min_distance).abs() > 0.002 {
            min_distance = f32::INFINITY;

            let mut unique_edges: Vec<(i32, i32)> = vec![];

            let mut i :i32 = 0;
            while i < normals.len() as i32 {
                if normals[i as usize].0.dot(support) >= 0.0 {
                    let f = (i * 3) as usize;

                    add_if_unique_edges(&mut unique_edges, &faces, f,     f + 1);
                    add_if_unique_edges(&mut unique_edges, &faces, f + 1, f + 2);
                    add_if_unique_edges(&mut unique_edges, &faces, f + 2, f    );

                    faces[f + 2] = *faces.last().unwrap(); faces.pop();
                    faces[f + 1] = *faces.last().unwrap(); faces.pop();
                    faces[f    ] = *faces.last().unwrap(); faces.pop();

                    normals[i as usize] = *normals.last().unwrap(); normals.pop();

                    i-=1;
                }
                i+=1;
            }

            if unique_edges.len() == 0 {
                break;
            }

            let mut new_faces = vec![];
            for (edge_1, edge_2) in unique_edges {
                new_faces.push(edge_1);
                new_faces.push(edge_2);
                new_faces.push(polytope.points.len() as i32);
            }

            polytope.points.push(support);

            let (new_normals, new_min_face) = get_face_normals(&polytope, &new_faces);

            let mut new_min_distance = f32::INFINITY;
            for i in 0..normals.len() {
                if normals[i].1 < new_min_distance {
                    new_min_distance = normals[i].1;
                    min_face = i as i32;
                }
            }

            if new_normals[new_min_face as usize].1 < new_min_distance {
                min_face = new_min_face + normals.len() as i32;
            }

            faces.extend(new_faces.iter().cloned());
            normals.extend(new_normals.iter().cloned());
        }
    }

    if min_distance == f32::INFINITY {
        return None;
    }

    let collision_vector = CollisionVector {
        normal: min_normal,
        depth: min_distance + 0.001,
    };

    return Some(collision_vector);
}

fn get_face_normals(polytope: &Polytope, faces: &Vec<i32>) -> (Vec<(cgmath::Vector3<f32>, f32)>, i32) {
    let mut normals: Vec<(Vector3<f32>, f32)> = vec![];
    let mut min_triangle = 0;
	let mut min_distance = f32::INFINITY;
    let mut index = 0;
    for chunk in faces.chunks(3) {
        let a = polytope.points[chunk[0] as usize];
        let b = polytope.points[chunk[1] as usize];
        let c = polytope.points[chunk[2] as usize];

        let mut normal = ((b - a).cross(c - a)).normalize();
        let mut distance = normal.dot(a);

        if distance < 0.0 {
            normal *= -1.0;
            distance *= -1.0;
        }

        normals.push((normal, distance));

        if distance < min_distance {
            min_triangle = index / 3;
            min_distance = distance;
        }
        index+=3;
    }

    return (normals, min_triangle);
}

fn add_if_unique_edges(unique_edges: &mut Vec<(i32, i32)>, faces: &Vec<i32>, first: usize, second: usize) {
    if let Some(reverse) = unique_edges.iter().position(|&(x, y)| x as i32 == faces[second] && y as i32 == faces[first]) {
        unique_edges.remove(reverse);
    } else {
        unique_edges.push((faces[first], faces[second]));
    }
}
