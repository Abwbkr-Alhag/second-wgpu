use cgmath::{Point3, Vector3};
use std::marker::PhantomData;
use std::ptr::NonNull;
use crate::aab::{AxisAlignedBounding};
use std::iter::IntoIterator;
use std::collections::{HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::fmt::Debug;

pub trait Collider {
    fn position(&self) -> Point3<f32>;
    fn scale(&self) -> Vector3<f32>;
    fn id(&self) -> usize;
    fn set_position(&mut self, new_position: Point3<f32>);
}


pub struct BVTree<T: AxisAlignedBounding<V>, V: Collider> {
    root: TreeNode<T, V>,
    size: usize,
}

type TreeNode<T, V> = Option<NonNull<Node<T, V>>>;

struct Wrapper<T: AxisAlignedBounding<V, Aab = T>, V: Collider>(TreeNode<T, V>);

// None of these methods check before accessing
impl<T: AxisAlignedBounding<V, Aab = T>, V: Collider> Wrapper<T, V> {
    fn parent(&self) -> TreeNode<T, V> {
        unsafe { (*self.0.unwrap().as_ptr()).parent }
    }
    fn grandparent(&self) -> TreeNode<T, V> {
        Wrapper(self.parent()).parent()
    }
    fn left_child(&self) -> TreeNode<T, V> {
        unsafe { (*self.0.unwrap().as_ptr()).left }
    }
    fn right_child(&self) -> TreeNode<T, V> {
        unsafe { (*self.0.unwrap().as_ptr()).right }
    }
    fn aab(&self) -> &T {
        unsafe { &(*self.0.unwrap().as_ptr()).aab }
    }
    fn collider(&self) -> &Option<V> {
        unsafe { &(*self.0.unwrap().as_ptr()).collider }
    }
    fn collider_id(&self) -> usize {
        unsafe { (*self.0.unwrap().as_ptr()).collider.as_ref().unwrap().id() }
    }
    fn child_is(&self) -> &ChildIs {
        unsafe { &(*self.0.unwrap().as_ptr()).child_is }
    }
    fn is_leaf(&self) -> bool {
        Wrapper(self.left_child()).0.is_none() && Wrapper(self.right_child()).0.is_none()
    }
    fn merge_aab(&self) {
        unsafe {
            (*self.0.unwrap().as_ptr()).aab = Wrapper(self.left_child()).aab().merge(&Wrapper(self.right_child()).aab());
        }
    }
}


#[derive(Clone, Debug)]
enum ChildIs {
    Left,
    Right,
    Null
}

struct Node<T: AxisAlignedBounding<V>, V: Collider> {
    parent: TreeNode<T, V>,
    left: TreeNode<T, V>,
    right: TreeNode<T, V>,
    aab: T,
    collider: Option<V>,
    child_is: ChildIs,
}

impl<T: AxisAlignedBounding<V>, V: Collider> Node<T, V> {
    fn is_leaf(&self) -> bool {
        // Left and Right should be none as there is either 0 or 2 children in a BVTree
        self.left.is_none() && self.right.is_none()
    }
}
impl<'a, T: AxisAlignedBounding<V, Aab = T> + std::fmt::Debug + Clone, V : Collider> BVTree<T, V> {
    pub fn new() -> Self {
        Self {
            root: None,
            size: 0,
        }
    }

    fn new_leaf_node(collider: V) -> TreeNode<T, V> {
        unsafe {
            Some(NonNull::new_unchecked(Box::into_raw(Box::new(Node {
                parent: None,
                left: None,
                right: None,
                aab: T::create_aab(&collider),
                collider: Some(collider),
                child_is: ChildIs::Null,
            }))))
        }
    }

    // left and right must exist in this function
    fn new_branch_node(&mut self, left: TreeNode<T, V>, right: TreeNode<T, V>) -> TreeNode<T, V> {
        unsafe {
            let child_is = &(*left.unwrap().as_ptr()).child_is.clone();
            let new_branch: TreeNode<T, V> = Some(NonNull::new_unchecked(Box::into_raw(Box::new(Node {
                parent: Wrapper(left).parent(),
                left,
                right,
                collider: None,
                aab: Wrapper(left).aab().merge(Wrapper(right).aab()),
                child_is: child_is.clone(),
            }))));
            if left == self.root || right == self.root {
                self.root = new_branch;
            }
            if Wrapper(new_branch).parent().is_some() {
                match child_is.clone() {
                    ChildIs::Left => {
                        (*Wrapper(new_branch).parent().unwrap().as_ptr()).left = new_branch;
                    },
                    ChildIs::Right => {
                        (*Wrapper(new_branch).parent().unwrap().as_ptr()).right = new_branch;
                    },
                    ChildIs::Null => {
                        panic!("Undefined");
                    }
                }
            }
            // Grandparent needs to have the respective left/right pointer towards child as well
            (*left.unwrap().as_ptr()).parent = new_branch;
            (*left.unwrap().as_ptr()).child_is = ChildIs::Left;
            (*right.unwrap().as_ptr()).parent = new_branch;
            (*right.unwrap().as_ptr()).child_is = ChildIs::Right;
            new_branch
        }
    }

    fn cost(child: TreeNode<T, V>, leaf: TreeNode<T, V>) -> f32 {
        Wrapper(child).aab().merge(Wrapper(leaf).aab()).surface_area_cost() -
        Wrapper(child).aab().surface_area_cost()
    }

    pub fn size(&self) -> usize {
        return self.size
    }

    pub fn len_branches(&self) -> usize {
        // Because of the structure of our tree we will always have
        // one less branch than we have leaves, feel free to test it out
        return self.size - 1;
    }

    pub fn insert(&mut self, collider: V) {
        let leaf = BVTree::new_leaf_node(collider);
        if self.root.is_some() {
            let mut node = self.root;
            loop {
                if Wrapper(node).is_leaf() {
                    self.new_branch_node(node, leaf);
                    // Node has become one of the leafs and we want to iterate on the branches
                    node = Wrapper(node).parent();
                    while node.is_some() {
                        Wrapper(node).merge_aab();
                        node = Wrapper(node).parent();
                    }
                    break;
                } else {
                    let left_cost = BVTree::cost(Wrapper(node).left_child(), leaf);
                    let right_cost = BVTree::cost(Wrapper(node).right_child(), leaf);
                    if left_cost < right_cost {
                        node = Wrapper(node).left_child();
                    } else {
                        node = Wrapper(node).right_child();
                    }
                }
            }
        } else {
            self.root = leaf;
        }
        self.size+=1;      
    }

    pub fn remove(&mut self, collider_id: usize, aab: &T) -> V {
        unsafe {
            // We want to deal with the edge cases that the node to remove is the root or a child of the root
            if self.root.is_none() {
                panic!("Removing a node that doesn't exist!");
            } else if Wrapper(self.root).collider().is_some() && Wrapper(self.root).collider_id() == collider_id {
                let node = Box::from_raw(self.root.unwrap().as_ptr()).collider.unwrap();
                self.root = None;
                self.size-=1;
                return node;
            } else if self.root.is_some() && Wrapper(Wrapper(self.root).left_child()).is_leaf() && Wrapper(Wrapper(self.root).left_child()).collider_id() == collider_id {
                let left = Box::from_raw(Wrapper(Wrapper(self.root).left_child()).0.unwrap().as_ptr());
                let right = Wrapper(self.root).right_child();
                drop(Box::from_raw(self.root.unwrap().as_ptr()));
                self.root = right;
                (*self.root.unwrap().as_ptr()).child_is = ChildIs::Null;
                (*self.root.unwrap().as_ptr()).parent = None;
                self.size-=1;
                return left.collider.unwrap();
            } else if self.root.is_some() && Wrapper(Wrapper(self.root).right_child()).is_leaf() && Wrapper(Wrapper(self.root).right_child()).collider_id() == collider_id {
                let right = Box::from_raw(Wrapper(Wrapper(self.root).right_child()).0.unwrap().as_ptr());
                let left = Wrapper(self.root).left_child();
                drop(Box::from_raw(self.root.unwrap().as_ptr()));
                self.root = left;
                (*self.root.unwrap().as_ptr()).child_is = ChildIs::Null;
                (*self.root.unwrap().as_ptr()).parent = None;
                self.size-=1;
                return right.collider.unwrap();
            }
    
            let mut stack = vec![self.root];
            while let Some(node) = stack.pop() {
                // This is the node we have to remove
                if Wrapper(node).is_leaf() {
                    if collider_id == Wrapper(node).collider_id() {
                        self.size-=1;
                        return BVTree::swap_out(node, Wrapper(node).parent(), Wrapper(node).grandparent());
                    } else {
                        continue;
                    }
                } else {
                    if aab.intersect(&Wrapper(Wrapper(node).left_child()).aab()) {
                        stack.push(Wrapper(node).left_child());
                    }
                    if aab.intersect(&Wrapper(Wrapper(node).right_child()).aab()) {
                        stack.push(Wrapper(node).right_child());
                    }
                }
    
            }
            panic!("Can't remove a node that doesn't exist!");
        }
    }
    
    pub fn update(&mut self, collider_id: usize, new_position: Point3<f32>, aab: &T) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let old_position = aab.position();
        let mut removed_collider = self.remove(collider_id, aab);
        (removed_collider).set_position(
            Point3 {
                x: old_position.x + rng.gen_range(-0.75..0.75),
                y: old_position.y + rng.gen_range(-0.75..0.75),
                z: old_position.z + rng.gen_range(-0.75..0.75),
            }               
        );
        // self.print_pretty("".to_string());
        self.insert(removed_collider);
        // self.print_pretty("".to_string());
    }

    pub fn get_branches<'b>(&'b self) -> Vec<&'b T> {
        let mut branches: Vec<&'b T> = vec![];
        if self.root.is_none() {
            return branches;
        }
        unsafe {
            let mut stack = vec![self.root];
            while let Some(node) = stack.pop() {
                if !Wrapper(node).is_leaf() {
                    branches.push(&(*node.unwrap().as_ptr()).aab);
                    stack.push(Wrapper(node).left_child());
                    stack.push(Wrapper(node).right_child());
                }
            }
        }
        return branches;
    }
    

    pub fn get_possible_pairs(&self) -> HashSet<ColliderPair<V>> {
        let mut set: HashSet<ColliderPair<V>> = HashSet::new();
        for inst in self.node_iter() {
            let collisions: Vec<(&V, &V)> = self.get_collisions(&inst);
            for collision in collisions {
                let pair = ColliderPair(collision.0, collision.1);
                set.insert(pair);
            }
        }
        return set;
    }

    pub fn point_intersection(&self, point: &Point3<f32>) -> Option<V> where V: Clone {
        if self.root.is_none() {
            return None;
        }
        let mut stack = vec![self.root];
        while let Some(node) = stack.pop() {
            if Wrapper(node).is_leaf() 
                && Wrapper(node).aab().point_intersects(&point) {
                return Some((Wrapper(node).collider().as_ref().unwrap()).clone());
            } 
            
            if Wrapper(Wrapper(node).left_child()).aab().point_intersects(&point) {
                stack.push(Wrapper(node).left_child());
            }
            if Wrapper(Wrapper(node).right_child()).aab().point_intersects(&point) {    
                stack.push(Wrapper(node).right_child());
            }
        }
        None
    }

    pub fn iter(&self) -> BVTreeIter<T, V> {
        BVTreeIter {
            queue: VecDeque::from([self.root]),
            _boo: PhantomData,
        }
    }

    pub fn iter_mut(&mut self) -> BVTreeIterMut<T, V> {
        BVTreeIterMut {
            queue: VecDeque::from([self.root]),
            _boo: PhantomData,
        }
    }

    pub fn into_iter(self) -> BVTreeIntoIter<T, V> {
        BVTreeIntoIter { 
            tree: self
        }
    }

    fn swap_out(child: TreeNode<T, V>, parent: TreeNode<T, V>, grand_parent: TreeNode<T, V>) -> V {
        unsafe {
            match Wrapper(child).child_is() {
                ChildIs::Left => {
                    match Wrapper(parent).child_is() {
                        ChildIs::Left => {
                            (*grand_parent.unwrap().as_ptr()).left = Wrapper(parent).right_child();
                            // Other child has their ChildIs Switched if it would be wrong in the new tree
                            (*Wrapper(parent).right_child().unwrap().as_ptr()).child_is = ChildIs::Left;
                        },
                        ChildIs::Right => {
                            (*grand_parent.unwrap().as_ptr()).right = Wrapper(parent).right_child();
                        },
                        ChildIs::Null => {
                            panic!("Shouldn't have a root node here!");
                        }
                    }
                    (*Wrapper(parent).right_child().unwrap().as_ptr()).parent = grand_parent;
                },
                ChildIs::Right => {
                    match Wrapper(parent).child_is() {
                        ChildIs::Left => {
                            (*grand_parent.unwrap().as_ptr()).left = Wrapper(parent).left_child();
                        }, 
                        ChildIs::Right => {
                            (*grand_parent.unwrap().as_ptr()).right = Wrapper(parent).left_child();
                            // Other child has their ChildIs Switched if it would be wrong in the new tree
                            (*Wrapper(parent).left_child().unwrap().as_ptr()).child_is = ChildIs::Right;
                        },
                        ChildIs::Null => {
                            panic!("Shouldn't have a root node here!");
                        }
                    }
                    (*Wrapper(parent).left_child().unwrap().as_ptr()).parent = grand_parent;
                },
                ChildIs::Null => {
                    panic!("Shouldn't have a root node here!");
                }
            }
            Wrapper(grand_parent).merge_aab();
            (*parent.unwrap().as_ptr()).parent = None;
            (*parent.unwrap().as_ptr()).left = None;
            (*parent.unwrap().as_ptr()).right = None;
            (*child.unwrap().as_ptr()).parent = None;
            drop(Box::from_raw(parent.unwrap().as_ptr()));
            return Box::from_raw(child.unwrap().as_ptr()).collider.unwrap();
        }
    }

    fn print_pretty(&self, indent: String) where T: AxisAlignedBounding<V, Aab = T>, V : Collider {
        self.print_pretty_helper(&self.root, indent);
    }

    fn print_pretty_helper(&self, tree: &TreeNode<T, V>, indent: String) where T: AxisAlignedBounding<V, Aab = T>, V : Collider {
        println!("{}+-AAB:{:?}, {:?}, volume: {}", indent, Wrapper(*tree).aab(), Wrapper(*tree).child_is(), Wrapper(*tree).aab().surface_area_cost());

        let mut str_ptr = indent.clone();
        if Wrapper(*tree).is_leaf() {
            str_ptr.push_str("   ");
        } else {
            str_ptr.push_str("|  ");
        }

        if !Wrapper(*tree).is_leaf() {
            self.print_pretty_helper(&Wrapper(*tree).left_child(), str_ptr.clone());
            self.print_pretty_helper(&Wrapper(*tree).right_child(), str_ptr);
        } else {
            return;
        }
    }

    fn get_collisions<'b>(&self, collision_node: &TreeNode<T, V>) -> Vec<(&'b V, &'b V)> {
        if self.root.is_none() {
            return vec![];
        }
        unsafe {
            let mut stack = vec![self.root];
            let mut node_vec: Vec<(&V, &V)> = vec![];
            let collider = (*collision_node.as_ref().unwrap().as_ptr()).collider.as_ref().unwrap();
            let collider_id = collider.id();
            while let Some(node) = stack.pop() {
                if Wrapper(node).is_leaf() {
                    let node_collider = (*node.unwrap().as_ptr()).collider.as_ref().unwrap();
                    if node_collider.id() != collider_id && Wrapper(node).aab().intersect(Wrapper(*collision_node).aab()) {
                        if node_collider.id() > collider_id {
                            node_vec.push((&collider, node_collider));
                        } else {
                            node_vec.push((node_collider, &collider));
                        }
                    }
                    continue;
                } 
                if Wrapper(Wrapper(node).left_child()).aab().intersect(Wrapper(*collision_node).aab()) {
                    stack.push(Wrapper(node).left_child());
                }
                if Wrapper(Wrapper(node).right_child()).aab().intersect(Wrapper(*collision_node).aab()) {    
                    stack.push(Wrapper(node).right_child())
                }
            }
            return node_vec;
        }
    }

    pub fn solve_collisions(&'a mut self) {
        let collision_id_vec: Vec<(usize, Point3<f32>, T)> = self.node_iter().collect::<Vec<Option<NonNull<Node<T, V>>>>>().iter().map(|node| {
            (Wrapper(*node).collider_id(), Wrapper(*node).collider().as_ref().unwrap().position(), (*Wrapper(*node).aab()).clone())
        }).collect();
        for collider in collision_id_vec {
            self.update(collider.0, collider.1, &collider.2);
        }
    }

    fn node_iter(&self) -> BVTreeNodeIter<T, V> {
        BVTreeNodeIter {
            queue: VecDeque::from([self.root]),
            _boo: PhantomData,
        }
    }
}

struct BVTreeNodeIter<'a, T: AxisAlignedBounding<V, Aab = T>, V : Collider> {
    queue: VecDeque<TreeNode<T, V>>,
    _boo: PhantomData<&'a TreeNode<T, V>>
}

impl<'a, T: AxisAlignedBounding<V, Aab = T>, V : Collider> Iterator for BVTreeNodeIter<'a, T, V> {
    type Item = Option<NonNull<Node<T, V>>>;
    
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.queue.pop_front() {
            unsafe {
                if !(*node.unwrap().as_ptr()).is_leaf() {
                    self.queue.push_back((*node.unwrap().as_ptr()).left);
                    self.queue.push_back((*node.unwrap().as_ptr()).right);
                } else {
                    return Some(node);
                }
            }
        }
        None
    }
}

pub struct BVTreeIter<'a, T: AxisAlignedBounding<V, Aab = T>, V : Collider> {
    queue: VecDeque<TreeNode<T, V>>,
    _boo: PhantomData<&'a TreeNode<T, V>>
}

impl<'a, T: AxisAlignedBounding<V, Aab = T>, V : Collider> Iterator for BVTreeIter<'a, T, V> {
    type Item = &'a V;
    
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.queue.pop_front() {
            unsafe {
                if !(*node.unwrap().as_ptr()).is_leaf() {
                    self.queue.push_back((*node.unwrap().as_ptr()).left);
                    self.queue.push_back((*node.unwrap().as_ptr()).right);
                } else {
                    return (*node.unwrap().as_ptr()).collider.as_ref();
                }
            }
        }

        None
    }
}
pub struct BVTreeIterMut<'a, T: AxisAlignedBounding<V, Aab = T>, V: Collider> {
    queue: VecDeque<TreeNode<T, V>>,
    _boo: PhantomData<&'a TreeNode<T, V>>
}

impl<'a, T: AxisAlignedBounding<V, Aab = T>, V: Collider> Iterator for BVTreeIterMut<'a, T, V> {
    type Item = &'a mut V;
    
    fn next(&mut self) -> Option<Self::Item> {

        while let Some(node) = self.queue.pop_front() {
            unsafe {
                if !(*node.unwrap().as_ptr()).is_leaf() {
                    self.queue.push_back((*node.unwrap().as_ptr()).left);
                    self.queue.push_back((*node.unwrap().as_ptr()).right);
                } else {
                    return (*node.unwrap().as_ptr()).collider.as_mut();
                }
            }
        }

        None
    }
}


pub struct BVTreeIntoIter<T: AxisAlignedBounding<V, Aab = T>, V: Collider> {
    tree: BVTree<T, V>,
}

impl<'a, T: AxisAlignedBounding<V, Aab = T>, V: Collider> IntoIterator for BVTree<T, V> {
    type IntoIter = BVTreeIntoIter<T, V>;
    type Item = V;

    fn into_iter(self) -> Self::IntoIter {
        BVTreeIntoIter { tree: self }
    }
}

impl<'a, T: AxisAlignedBounding<V, Aab = T>, V: Collider> Iterator for BVTreeIntoIter<T, V> {
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
        // let next = self.tree.iter().next();
        // if let Some(next) = next {
        //     self.tree.remove(next);
        //     Some()
        // } else {
        //     None
        // }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.tree.size as usize, Some(self.tree.size as usize))
    }
}

impl<'a, T: AxisAlignedBounding<V, Aab = T> + std::fmt::Debug + Clone, V: Collider> IntoIterator for &'a BVTree<T, V> {
    type Item = &'a V;
    type IntoIter = BVTreeIter<'a, T, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, V> Extend<V> for BVTree<T, V> where T: AxisAlignedBounding<V, Aab = T> + std::fmt::Debug + Clone, V : Collider {
    fn extend<I: IntoIterator<Item = V>>(&mut self, iter: I) {
        for item in iter {
            self.insert(item);
        }
    }
}

use std::fmt;
impl<T, V> Debug for BVTree<T, V> where T: AxisAlignedBounding<V, Aab = T> + std::fmt::Debug + Clone, V : Collider + Debug {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(self.print_pretty("".to_string()))
    }
}

pub struct ColliderPair<'a, V: Collider>(&'a V, &'a V);

impl<'a, V: Collider> Hash for ColliderPair<'a, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let (a, b) = (self.clone().0, self.clone().1);
        a.id().hash(state);
        b.id().hash(state);
    }
}

impl<'a, V: Collider> PartialEq for ColliderPair<'a, V> {
    fn eq(&self, other: &Self) -> bool {
        let (a1, b1) = (self.clone().0, self.clone().1);
        let (a2, b2) = (other.clone().0, other.clone().1);
        a1.id() == a2.id() && b1.id() == b2.id()
    }
}

impl<'a, V: Collider> Clone for ColliderPair<'a, V> {
    fn clone(&self) -> Self {
        ColliderPair(self.0, self.1)
    }
}

impl<'a, V: Collider> Eq for ColliderPair<'a, V> {}

impl<'a, V: Collider + std::fmt::Debug> Debug for ColliderPair<'a, V> {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(println!("{:?}, {:?}", self.0, self.1))
    }
}


#[cfg(test)]
mod test {
    use super::BVTree;
    use crate::aab::{AABRect};
    use crate::Instance;
    use cgmath::Rotation3;

    #[test]
    fn test_basic_insert_and_delete() {
        let mut m: BVTree<AABRect, Instance> = BVTree::new();
        for i in 1..4 {
            m.insert(Instance::new(
                cgmath::Point3 {
                    x: -1.0 * i as f32,
                    y: -1.0 * i as f32,
                    z: -1.0 * i as f32,
                },
                {
                    cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
                },
                cgmath::Vector3 {
                    x: 0.5 * i as f32,
                    y: 0.5 * i as f32,
                    z: 0.5 * i as f32,
                },
            ));
        }
        let size = m.size();
        println!("{:?}", m);
        m.solve_collisions();
        assert!(size == m.size())
    }

    #[test]
    fn test_large_insert_and_delete() {
        let mut m: BVTree<AABRect, Instance> = BVTree::new();
        for i in 1..50 {
            m.insert(Instance::new(
                cgmath::Point3 {
                    x: -1.0 * i as f32,
                    y: -1.0 * i as f32,
                    z: -1.0 * i as f32,
                },
                {
                    cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
                },
                cgmath::Vector3 {
                    x: 0.5 * i as f32,
                    y: 0.5 * i as f32,
                    z: 0.5 * i as f32,
                },
            ));
        }
        let size = m.size();
        println!("{:?}", m);
        m.solve_collisions();
        assert!(size == m.size())
    }
}
