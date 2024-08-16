// SPDX-License-Identifier: GPL-2.0

#include <linux/slab.h>
// TODO: Move vrealloc() to a different file or rename this file.
#include <linux/vmalloc.h>

void * __must_check __realloc_size(2)
rust_helper_krealloc(const void *objp, size_t new_size, gfp_t flags)
{
	return krealloc(objp, new_size, flags);
}

// TODO: Move vrealloc() to a different file or rename this file.
void * __must_check __realloc_size(2)
rust_helper_vrealloc(const void *p, size_t size, gfp_t flags)
{
	return vrealloc(p, size, flags);
}

void * __must_check __realloc_size(2)
rust_helper_kvrealloc(const void *p, size_t size, gfp_t flags)
{
	return kvrealloc(p, size, flags);
}
