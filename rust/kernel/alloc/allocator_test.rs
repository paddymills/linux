// SPDX-License-Identifier: GPL-2.0

#![allow(missing_docs)]

use super::{flags::*, AllocError, Allocator, Flags};
use core::alloc::Layout;
use core::cmp;
use core::mem;
use core::ptr;
use core::ptr::NonNull;

pub struct Cmalloc;
pub type Kmalloc = Cmalloc;
pub type Vmalloc = Kmalloc;
pub type KVmalloc = Kmalloc;

extern "C" {
    #[link_name = "aligned_alloc"]
    fn libc_aligned_alloc(align: usize, size: usize) -> *mut core::ffi::c_void;

    #[link_name = "free"]
    fn libc_free(ptr: *mut core::ffi::c_void);
}

struct CmallocData {
    // The actual size as requested through `Cmalloc::alloc` or `Cmalloc::realloc`.
    size: usize,
    // The offset from the pointer returned to the caller of `Cmalloc::alloc` or `Cmalloc::realloc`
    // to the actual base address of the allocation.
    offset: usize,
}

impl Cmalloc {
    /// Adjust the size and alignment such that we can additionally store `CmallocData` right
    /// before the actual data described by `layout`.
    ///
    /// Example:
    ///
    /// For `CmallocData` assume an alignment of 8 and a size of 16.
    /// For `layout` assume and alignment of 16 and a size of 64.
    ///
    /// 0                16               32                                               96
    /// |----------------|----------------|------------------------------------------------|
    ///        empty         CmallocData                         data
    ///
    /// For this example the returned `Layout` has an alignment of 32 and a size of 96.
    fn layout_adjust(layout: Layout) -> Result<Layout, AllocError> {
        let layout = layout.pad_to_align();

        // Ensure that `CmallocData` fits into half the alignment. Additionally, this guarantees
        // that advancing a pointer aligned to `align` by `align / 2` we still satisfy or exceed
        // the alignment requested through `layout`.
        let align = cmp::max(
            layout.align(),
            mem::size_of::<CmallocData>().next_power_of_two(),
        ) * 2;

        // Add the additional space required for `CmallocData`.
        let size = layout.size() + mem::size_of::<CmallocData>();

        Ok(Layout::from_size_align(size, align)
            .map_err(|_| AllocError)?
            .pad_to_align())
    }

    fn alloc_store_data(layout: Layout) -> Result<NonNull<u8>, AllocError> {
        let requested_size = layout.size();

        let layout = Self::layout_adjust(layout)?;
        let min_align = layout.align() / 2;

        // SAFETY: Returns either NULL or a pointer to a memory allocation that satisfies or
        // exceeds the given size and alignment requirements.
        let raw_ptr = unsafe { libc_aligned_alloc(layout.align(), layout.size()) } as *mut u8;

        let priv_ptr = NonNull::new(raw_ptr).ok_or(AllocError)?;

        // SAFETY: Advance the pointer by `min_align`. The adjustments from `Self::layout_adjust`
        // ensure that after this operation the original size and alignment requirements are still
        // satisfied or exceeded.
        let ptr = unsafe { priv_ptr.as_ptr().add(min_align) };

        // SAFETY: `min_align` is greater than or equal to the size of `CmallocData`, hence we
        // don't exceed the allocation boundaries.
        let data_ptr: *mut CmallocData = unsafe { ptr.sub(mem::size_of::<CmallocData>()) }.cast();

        let data = CmallocData {
            size: requested_size,
            offset: min_align,
        };

        // SAFETY: `data_ptr` is properly aligned and within the allocation boundaries reserved for
        // `CmallocData`.
        unsafe { data_ptr.write(data) };

        NonNull::new(ptr).ok_or(AllocError)
    }

    /// # Safety
    ///
    /// `ptr` must have been previously allocated with `Self::alloc_store_data`.
    unsafe fn data<'a>(ptr: NonNull<u8>) -> &'a CmallocData {
        // SAFETY: `Self::alloc_store_data` stores the `CmallocData` right before the address
        // returned to callers of `Self::alloc_store_data`.
        let data_ptr: *mut CmallocData =
            unsafe { ptr.as_ptr().sub(mem::size_of::<CmallocData>()) }.cast();

        // SAFETY: The `CmallocData` has been previously stored at this offset with
        // `Self::alloc_store_data`.
        unsafe { &*data_ptr }
    }

    /// # Safety
    ///
    /// This function must not be called more than once for the same allocation.
    ///
    /// `ptr` must have been previously allocated with `Self::alloc_store_data`.
    unsafe fn free_read_data(ptr: NonNull<u8>) {
        // SAFETY: `ptr` has been created by `Self::alloc_store_data`.
        let data = unsafe { Self::data(ptr) };

        // SAFETY: `ptr` has been created by `Self::alloc_store_data`.
        let priv_ptr = unsafe { ptr.as_ptr().sub(data.offset) };

        // SAFETY: `priv_ptr` has previously been allocatored with this `Allocator`.
        unsafe { libc_free(priv_ptr.cast()) };
    }
}

unsafe impl Allocator for Cmalloc {
    fn alloc(layout: Layout, flags: Flags) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(NonNull::slice_from_raw_parts(NonNull::dangling(), 0));
        }

        let ptr = Self::alloc_store_data(layout)?;

        if flags.contains(__GFP_ZERO) {
            // SAFETY: `Self::alloc_store_data` guarantees that `ptr` points to memory of at least
            // `layout.size()` bytes.
            unsafe { ptr.as_ptr().write_bytes(0, layout.size()) };
        }

        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }

    unsafe fn realloc(
        ptr: Option<NonNull<u8>>,
        layout: Layout,
        flags: Flags,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let src: NonNull<u8> = if let Some(src) = ptr {
            src.cast()
        } else {
            return Self::alloc(layout, flags);
        };

        if layout.size() == 0 {
            // SAFETY: `src` has been created by `Self::alloc_store_data`.
            unsafe { Self::free_read_data(src) };

            return Ok(NonNull::slice_from_raw_parts(NonNull::dangling(), 0));
        }

        let dst = Self::alloc(layout, flags)?;

        // SAFETY: `src` has been created by `Self::alloc_store_data`.
        let data = unsafe { Self::data(src) };

        // SAFETY: `src` has previously been allocated with this `Allocator`; `dst` has just been
        // newly allocated. Copy up to the smaller of both sizes.
        unsafe {
            ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.as_ptr().cast(),
                cmp::min(layout.size(), data.size),
            )
        };

        // SAFETY: `src` has been created by `Self::alloc_store_data`.
        unsafe { Self::free_read_data(src) };

        Ok(dst)
    }
}
