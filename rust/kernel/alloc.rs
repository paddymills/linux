// SPDX-License-Identifier: GPL-2.0

//! Extensions to the [`alloc`] crate.

#[cfg(not(any(test, testlib)))]
pub mod allocator;
pub mod box_ext;
pub mod kbox;
pub mod vec_ext;

#[cfg(any(test, testlib))]
pub mod allocator_test;

#[cfg(any(test, testlib))]
pub use self::allocator_test as allocator;

pub use self::kbox::Box;
pub use self::kbox::KBox;
pub use self::kbox::KVBox;
pub use self::kbox::VBox;

/// Indicates an allocation error.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct AllocError;
use core::{alloc::Layout, ptr::NonNull};

/// Flags to be used when allocating memory.
///
/// They can be combined with the operators `|`, `&`, and `!`.
///
/// Values can be used from the [`flags`] module.
#[derive(Clone, Copy)]
pub struct Flags(u32);

impl Flags {
    /// Get the raw representation of this flag.
    pub(crate) fn as_raw(self) -> u32 {
        self.0
    }
}

impl core::ops::BitOr for Flags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitAnd for Flags {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl core::ops::Not for Flags {
    type Output = Self;
    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

/// Allocation flags.
///
/// These are meant to be used in functions that can allocate memory.
pub mod flags {
    use super::Flags;

    /// Zeroes out the allocated memory.
    ///
    /// This is normally or'd with other flags.
    pub const __GFP_ZERO: Flags = Flags(bindings::__GFP_ZERO);

    /// Allow the allocation to be in high memory.
    ///
    /// Allocations in high memory may not be mapped into the kernel's address space, so this can't
    /// be used with `kmalloc` and other similar methods.
    ///
    /// This is normally or'd with other flags.
    pub const __GFP_HIGHMEM: Flags = Flags(bindings::__GFP_HIGHMEM);

    /// Users can not sleep and need the allocation to succeed.
    ///
    /// A lower watermark is applied to allow access to "atomic reserves". The current
    /// implementation doesn't support NMI and few other strict non-preemptive contexts (e.g.
    /// raw_spin_lock). The same applies to [`GFP_NOWAIT`].
    pub const GFP_ATOMIC: Flags = Flags(bindings::GFP_ATOMIC);

    /// Typical for kernel-internal allocations. The caller requires ZONE_NORMAL or a lower zone
    /// for direct access but can direct reclaim.
    pub const GFP_KERNEL: Flags = Flags(bindings::GFP_KERNEL);

    /// The same as [`GFP_KERNEL`], except the allocation is accounted to kmemcg.
    pub const GFP_KERNEL_ACCOUNT: Flags = Flags(bindings::GFP_KERNEL_ACCOUNT);

    /// For kernel allocations that should not stall for direct reclaim, start physical IO or
    /// use any filesystem callback.  It is very likely to fail to allocate memory, even for very
    /// small allocations.
    pub const GFP_NOWAIT: Flags = Flags(bindings::GFP_NOWAIT);

    /// Suppresses allocation failure reports.
    ///
    /// This is normally or'd with other flags.
    pub const __GFP_NOWARN: Flags = Flags(bindings::__GFP_NOWARN);
}

/// The kernel's [`Allocator`] trait.
///
/// An implementation of [`Allocator`] can allocate, re-allocate and free memory buffer described
/// via [`Layout`].
///
/// [`Allocator`] is designed to be implemented as a ZST; [`Allocator`] functions do not operate on
/// an object instance.
///
/// In order to be able to support `#[derive(SmartPointer)]` later on, we need to avoid a design
/// that requires an `Allocator` to be instantiated, hence its functions must not contain any kind
/// of `self` parameter.
///
/// # Safety
///
/// A memory allocation returned from an allocator must remain valid until it is explicitly freed.
///
/// Any pointer to a valid memory allocation must be valid to be passed to any other [`Allocator`]
/// function of the same type.
///
/// Implementers must ensure that all trait functions abide by the guarantees documented in the
/// `# Guarantees` sections.
pub unsafe trait Allocator {
    /// Allocate memory based on `layout` and `flags`.
    ///
    /// On success, returns a buffer represented as `NonNull<[u8]>` that satisfies the layout
    /// constraints (i.e. minimum size and alignment as specified by `layout`).
    ///
    /// This function is equivalent to `realloc` when called with `None`.
    ///
    /// # Guarantees
    ///
    /// When the return value is `Ok(ptr)`, then `ptr` is
    /// - valid for reads and writes for `layout.size()` bytes, until it is passed to
    ///   [`Allocator::free`] or [`Allocator::realloc`],
    /// - aligned to `layout.align()`,
    ///
    /// Additionally, `Flags` are honored as documented in
    /// <https://docs.kernel.org/core-api/mm-api.html#mm-api-gfp-flags>.
    fn alloc(layout: Layout, flags: Flags) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: Passing `None` to `realloc` is valid by it's safety requirements and asks for a
        // new memory allocation.
        unsafe { Self::realloc(None, layout, flags) }
    }

    /// Re-allocate an existing memory allocation to satisfy the requested `layout`.
    ///
    /// If the requested size is zero, `realloc` behaves equivalent to `free`.
    ///
    /// If the requested size is larger than the size of the existing allocation, a successful call
    /// to `realloc` guarantees that the new or grown buffer has at least `Layout::size` bytes, but
    /// may also be larger.
    ///
    /// If the requested size is smaller than the size of the existing allocation, `realloc` may or
    /// may not shrink the buffer; this is implementation specific to the allocator.
    ///
    /// On allocation failure, the existing buffer, if any, remains valid.
    ///
    /// The buffer is represented as `NonNull<[u8]>`.
    ///
    /// # Safety
    ///
    /// If `ptr == Some(p)`, then `p` must point to an existing and valid memory allocation created
    /// by this allocator. The alignment encoded in `layout` must be smaller than or equal to the
    /// alignment requested in the previous `alloc` or `realloc` call of the same allocation.
    ///
    /// Additionally, `ptr` is allowed to be `None`; in this case a new memory allocation is
    /// created.
    ///
    /// # Guarantees
    ///
    /// This function has the same guarantees as [`Allocator::alloc`]. When `ptr == Some(p)`, then
    /// it additionally guarantees that:
    /// - the contents of the memory pointed to by `p` are preserved up to the lesser of the new
    ///   and old size,
    ///   and old size, i.e.
    ///   `ret_ptr[0..min(layout.size(), old_size)] == p[0..min(layout.size(), old_size)]`, where
    ///   `old_size` is the size of the allocation that `p` points at.

    /// - when the return value is `Err(AllocError)`, then `p` is still valid.
    unsafe fn realloc(
        ptr: Option<NonNull<u8>>,
        layout: Layout,
        flags: Flags,
    ) -> Result<NonNull<[u8]>, AllocError>;

    /// Free an existing memory allocation.
    ///
    /// # Safety
    ///
    /// `ptr` must point to an existing and valid memory allocation created by this `Allocator` and
    /// must not be a dangling pointer.
    ///
    /// The memory allocation at `ptr` must never again be read from or written to.
    unsafe fn free(ptr: NonNull<u8>) {
        // SAFETY: The caller guarantees that `ptr` points at a valid allocation created by this
        // allocator. We are passing a `Layout` with the smallest possible alignment, so it is
        // smaller than or equal to the alignment previously used with this allocation.
        let _ = unsafe { Self::realloc(Some(ptr), Layout::new::<()>(), Flags(0)) };
    }
}
