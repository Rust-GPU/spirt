//! Traversal helpers for intra-function entities.
//!
//! [`FuncAt<P>`]/[`FuncAtMut<P>`] are like `(&FuncDefBody, P)`/`(&mut FuncDefBody, P`)
//! (where `P` is some type describing a "position" in the function), except:
//! * they only borrow the [`EntityDefs`] fields of [`FuncDefBody`]
//!   * this can prevent borrow conflicts, especially when mutating other fields
//!   * it also avoids accidentally accessing parts of the function definition
//!     without going through `P` (as [`EntityDefs`] requires keys for any access)
//! * they're dedicated types with inherent methods and trait `impl`s

// NOTE(eddyb) wrong wrt lifetimes (https://github.com/rust-lang/rust-clippy/issues/5004).
#![allow(clippy::should_implement_trait)]

use crate::{
    Context, EntityDefs, EntityList, EntityListIter, FuncDefBody, Node, NodeDef, Region, RegionDef,
    Type, Value,
};

/// Immutable traversal (i.e. visiting) helper for intra-function entities.
///
/// The point/position type `P` should be an entity or a shallow entity wrapper
/// (e.g. [`EntityList<Node>`]).
#[derive(Copy, Clone)]
pub struct FuncAt<'a, P: Copy> {
    pub regions: &'a EntityDefs<Region>,
    pub nodes: &'a EntityDefs<Node>,

    pub position: P,
}

impl<'a, P: Copy> FuncAt<'a, P> {
    /// Reposition to `new_position`.
    pub fn at<P2: Copy>(self, new_position: P2) -> FuncAt<'a, P2> {
        FuncAt { regions: self.regions, nodes: self.nodes, position: new_position }
    }
}

impl<'a> FuncAt<'a, Region> {
    pub fn def(self) -> &'a RegionDef {
        &self.regions[self.position]
    }

    pub fn at_children(self) -> FuncAt<'a, EntityList<Node>> {
        self.at(self.def().children)
    }
}

impl<'a> IntoIterator for FuncAt<'a, EntityList<Node>> {
    type IntoIter = FuncAt<'a, EntityListIter<Node>>;
    type Item = FuncAt<'a, Node>;
    fn into_iter(self) -> Self::IntoIter {
        self.at(self.position.iter())
    }
}

impl<'a> Iterator for FuncAt<'a, EntityListIter<Node>> {
    type Item = FuncAt<'a, Node>;
    fn next(&mut self) -> Option<Self::Item> {
        let (next, rest) = self.position.split_first(self.nodes)?;
        self.position = rest;
        Some(self.at(next))
    }
}

impl DoubleEndedIterator for FuncAt<'_, EntityListIter<Node>> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let (prev, rest) = self.position.split_last(self.nodes)?;
        self.position = rest;
        Some(self.at(prev))
    }
}

impl<'a> FuncAt<'a, Node> {
    pub fn def(self) -> &'a NodeDef {
        &self.nodes[self.position]
    }
}

impl FuncAt<'_, Value> {
    /// Return the [`Type`] of this [`Value`] ([`Context`] used for [`Value::Const`]).
    pub fn type_of(self, cx: &Context) -> Type {
        match self.position {
            Value::Const(ct) => cx[ct].ty,
            Value::RegionInput { region, input_idx } => {
                self.at(region).def().inputs[input_idx as usize].ty
            }
            Value::NodeOutput { node, output_idx } => {
                self.at(node).def().outputs[output_idx as usize].ty
            }
            Value::DataInstOutput { inst, output_idx } => {
                self.at(inst).def().outputs[output_idx as usize].ty
            }
        }
    }
}

/// Mutable traversal (i.e. transforming) helper for intra-function entities.
///
/// The point/position type `P` should be an entity or a shallow entity wrapper
/// (e.g. [`EntityList<Node>`]).
pub struct FuncAtMut<'a, P: Copy> {
    pub regions: &'a mut EntityDefs<Region>,
    pub nodes: &'a mut EntityDefs<Node>,

    pub position: P,
}

impl<'a, P: Copy> FuncAtMut<'a, P> {
    /// Emulate a "reborrow", which is automatic only for `&mut` types.
    pub fn reborrow(&mut self) -> FuncAtMut<'_, P> {
        FuncAtMut { regions: self.regions, nodes: self.nodes, position: self.position }
    }

    /// Reposition to `new_position`.
    pub fn at<P2: Copy>(self, new_position: P2) -> FuncAtMut<'a, P2> {
        FuncAtMut { regions: self.regions, nodes: self.nodes, position: new_position }
    }

    /// Demote to a `FuncAt`, with the same `position`.
    //
    // FIXME(eddyb) maybe find a better name for this?
    pub fn freeze(self) -> FuncAt<'a, P> {
        let FuncAtMut { regions, nodes, position } = self;
        FuncAt { regions, nodes, position }
    }
}

impl<'a> FuncAtMut<'a, Region> {
    pub fn def(self) -> &'a mut RegionDef {
        &mut self.regions[self.position]
    }

    pub fn at_children(mut self) -> FuncAtMut<'a, EntityList<Node>> {
        let children = self.reborrow().def().children;
        self.at(children)
    }
}

// HACK(eddyb) can't implement `IntoIterator` because `next` borrows `self`.
impl<'a> FuncAtMut<'a, EntityList<Node>> {
    pub fn into_iter(self) -> FuncAtMut<'a, EntityListIter<Node>> {
        let iter = self.position.iter();
        self.at(iter)
    }
}

// HACK(eddyb) can't implement `Iterator` because `next` borrows `self`.
impl FuncAtMut<'_, EntityListIter<Node>> {
    pub fn next(&mut self) -> Option<FuncAtMut<'_, Node>> {
        let (next, rest) = self.position.split_first(self.nodes)?;
        self.position = rest;
        Some(self.reborrow().at(next))
    }
}

impl<'a> FuncAtMut<'a, Node> {
    pub fn def(self) -> &'a mut NodeDef {
        &mut self.nodes[self.position]
    }
}

impl FuncDefBody {
    /// Start immutably traversing the function at `position`.
    pub fn at<P: Copy>(&self, position: P) -> FuncAt<'_, P> {
        FuncAt { regions: &self.regions, nodes: &self.nodes, position }
    }

    /// Start mutably traversing the function at `position`.
    pub fn at_mut<P: Copy>(&mut self, position: P) -> FuncAtMut<'_, P> {
        FuncAtMut { regions: &mut self.regions, nodes: &mut self.nodes, position }
    }

    /// Shorthand for `func_def_body.at(func_def_body.body)`.
    pub fn at_body(&self) -> FuncAt<'_, Region> {
        self.at(self.body)
    }

    /// Shorthand for `func_def_body.at_mut(func_def_body.body)`.
    pub fn at_mut_body(&mut self) -> FuncAtMut<'_, Region> {
        self.at_mut(self.body)
    }
}
