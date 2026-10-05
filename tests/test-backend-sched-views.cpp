// Checks that ggml_backend_sched copies a tensor across a device boundary once per graph compute, even when several
// consumers on the far side read it through their own separately built - but identical - views.
//
// This is the shape llama.cpp produces whenever one tensor is attended by many later layers (e.g. a shared KV cache
// entry: every consuming layer calls get_k(il) and gets a distinct ggml_tensor that addresses the same bytes).
// Without deduplication the scheduler creates one copy tensor per consuming split and moves the same bytes once per
// split.
//
// The test needs two backends whose buffer types are mutually unsupported, which a CPU-only build does not have, so
// it defines a dummy "device" backend: plain host memory that reports is_host = false (so the CPU backend cannot use
// its buffers in place) and whose graph_compute delegates to a private CPU backend instance. Data leaving that device
// therefore goes through its buffer's get_tensor, which is counted.

#include "../ggml/src/ggml-backend-impl.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

//
// dummy device backend: host memory that pretends not to be host memory
//

struct dummy_dev {
    ggml_backend_buffer_type buft;
    ggml_backend_device      device;
    ggml_backend             backend;
    ggml_backend_t           cpu = nullptr; // does the actual computing

    size_t n_reads  = 0; // get_tensor calls on this device's buffers (bytes leaving the device)
    size_t n_writes = 0; // set_tensor calls
};

struct dummy_buffer {
    dummy_dev * dev;
    void *      raw;  // as returned by calloc
    void *      base; // raw, aligned up to the buffer type's alignment
};

// buffer

static void * dummy_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((dummy_buffer *) buffer->context)->base;
}

static void dummy_buffer_free(ggml_backend_buffer_t buffer) {
    dummy_buffer * b = (dummy_buffer *) buffer->context;
    free(b->raw);
    delete b;
}

static void dummy_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ((dummy_buffer *) buffer->context)->dev->n_writes++;
    memcpy((char *) tensor->data + offset, data, size);
}

static void dummy_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ((dummy_buffer *) buffer->context)->dev->n_reads++;
    memcpy(data, (const char *) tensor->data + offset, size);
}

static void dummy_buffer_memset_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);
}

static void dummy_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(((dummy_buffer *) buffer->context)->base, value, buffer->size);
}

// buffer type

static const char * dummy_buft_get_name(ggml_backend_buffer_type_t) {
    return "DUMMY";
}

static size_t dummy_buft_get_alignment(ggml_backend_buffer_type_t) {
    return 64;
}

static ggml_backend_buffer_t dummy_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t align = dummy_buft_get_alignment(buft);

    dummy_buffer * b = new dummy_buffer;
    b->dev  = (dummy_dev *) buft->device->context;
    b->raw  = calloc(1, size + align);
    GGML_ASSERT(b->raw);
    b->base = (void *) (((uintptr_t) b->raw + align - 1) & ~(uintptr_t) (align - 1));

    ggml_backend_buffer_i iface = {};
    iface.free_buffer   = dummy_buffer_free;
    iface.get_base      = dummy_buffer_get_base;
    iface.memset_tensor = dummy_buffer_memset_tensor;
    iface.set_tensor    = dummy_buffer_set_tensor;
    iface.get_tensor    = dummy_buffer_get_tensor;
    iface.clear         = dummy_buffer_clear;
    // note: no is_host on the buffer type and no cpy_tensor here, so every transfer out of this device is a
    // get_tensor call that the counters above see

    return ggml_backend_buffer_init(buft, iface, b, size);
}

// device

static const char * dummy_dev_get_name(ggml_backend_dev_t) {
    return "DUMMY";
}

static const char * dummy_dev_get_description(ggml_backend_dev_t) {
    return "dummy non-host device backed by the CPU";
}

static void dummy_dev_get_memory(ggml_backend_dev_t, size_t * free_, size_t * total) {
    *free_ = 1024 * 1024 * 1024;
    *total = 1024 * 1024 * 1024;
}

static enum ggml_backend_dev_type dummy_dev_get_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static ggml_backend_buffer_type_t dummy_dev_get_buffer_type(ggml_backend_dev_t dev) {
    return &((dummy_dev *) dev->context)->buft;
}

static bool dummy_dev_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return ggml_backend_supports_op(((dummy_dev *) dev->context)->cpu, op);
}

static bool dummy_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == &((dummy_dev *) dev->context)->buft;
}

static void dummy_dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    memset(props, 0, sizeof(*props));
    props->name        = dummy_dev_get_name(dev);
    props->description = dummy_dev_get_description(dev);
    props->type        = dummy_dev_get_type(dev);
    dummy_dev_get_memory(dev, &props->memory_free, &props->memory_total);
}

static ggml_backend_t dummy_dev_init_backend(ggml_backend_dev_t dev, const char *) {
    return &((dummy_dev *) dev->context)->backend;
}

// backend

static const char * dummy_backend_get_name(ggml_backend_t) {
    return "DUMMY";
}

static void dummy_backend_free(ggml_backend_t) {}

static enum ggml_status dummy_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    return ggml_backend_graph_compute(((dummy_dev *) backend->context)->cpu, cgraph);
}

static dummy_dev * dummy_dev_init() {
    dummy_dev * d = new dummy_dev;

    d->cpu = ggml_backend_cpu_init();
    GGML_ASSERT(d->cpu);
    ggml_backend_cpu_set_n_threads(d->cpu, 1);

    d->device.iface                  = {};
    d->device.iface.get_name         = dummy_dev_get_name;
    d->device.iface.get_description  = dummy_dev_get_description;
    d->device.iface.get_memory       = dummy_dev_get_memory;
    d->device.iface.get_type         = dummy_dev_get_type;
    d->device.iface.get_props        = dummy_dev_get_props;
    d->device.iface.init_backend     = dummy_dev_init_backend;
    d->device.iface.get_buffer_type  = dummy_dev_get_buffer_type;
    d->device.iface.supports_op      = dummy_dev_supports_op;
    d->device.iface.supports_buft    = dummy_dev_supports_buft;
    d->device.reg                    = nullptr;
    d->device.context                = d;

    d->buft.iface                = {};
    d->buft.iface.get_name       = dummy_buft_get_name;
    d->buft.iface.alloc_buffer   = dummy_buft_alloc_buffer;
    d->buft.iface.get_alignment  = dummy_buft_get_alignment;
    d->buft.device               = &d->device;
    d->buft.context              = d;

    d->backend.guid    = nullptr;
    d->backend.iface   = {};
    d->backend.iface.get_name      = dummy_backend_get_name;
    d->backend.iface.free          = dummy_backend_free;
    d->backend.iface.graph_compute = dummy_backend_graph_compute;
    d->backend.device  = &d->device;
    d->backend.context = d;

    return d;
}

static void dummy_dev_free(dummy_dev * d) {
    ggml_backend_free(d->cpu);
    delete d;
}

//
// test graphs
//

static const int64_t NE = 64; // rows
static const int64_t NC = 8;  // columns of the shared tensor
static const int     K  = 6;  // number of consumers on the far side

static void fill_f32(ggml_tensor * t, uint32_t seed) {
    std::vector<float> v(ggml_nelements(t));
    uint32_t s = seed * 2654435761u + 1;
    for (auto & x : v) {
        s = s * 1664525u + 1013904223u;
        x = ((s >> 8) & 0xffff) / 65535.0f - 0.5f;
    }
    ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
}

enum case_kind {
    CASE_SAME_VIEWS,  // K consumers, all reading the same view of the shared tensor
    CASE_DISJOINT,    // K consumers, each reading a different column of it (control: no deduplication possible)
    CASE_MODIFIED,    // two consumers of the same view, with an in-place write to the base in between
    CASE_RESHAPE,     // two consumers of the same bytes through differently shaped contiguous views
};

// builds and runs one case. `far` holds the weights of the ops that must run away from the dummy device (in a plain
// CPU buffer, which the dummy device does not support), `near` the ones that must run on it. When `dev` is null
// everything runs on one CPU backend, which is the reference.
static std::vector<float> run_case(case_kind kind, dummy_dev * dev, ggml_backend_t cpu, size_t * n_reads) {
    ggml_backend_t near_backend = dev ? &dev->backend : cpu;

    ggml_init_params ip = { ggml_tensor_overhead() * 64, nullptr, true };

    // weights of the ops that anchor to the dummy device
    ggml_context * ctx_near = ggml_init(ip);
    ggml_tensor * w_src = ggml_new_tensor_2d(ctx_near, GGML_TYPE_F32, NE, NE);
    ggml_tensor * w_mod = ggml_new_tensor_2d(ctx_near, GGML_TYPE_F32, NE, NE);
    ggml_tensor * w_near[K];
    for (int k = 0; k < K; k++) {
        w_near[k] = ggml_new_tensor_2d(ctx_near, GGML_TYPE_F32, NE, NE);
    }
    ggml_tensor * x = ggml_new_tensor_2d(ctx_near, GGML_TYPE_F32, NE, NC);
    ggml_backend_buffer_t buf_near = ggml_backend_alloc_ctx_tensors(ctx_near, near_backend);
    ggml_backend_buffer_set_usage(buf_near, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    fill_f32(w_src, 1);
    fill_f32(w_mod, 2);
    for (int k = 0; k < K; k++) {
        fill_f32(w_near[k], 10 + k);
    }
    fill_f32(x, 3);

    // weights of the ops that must run on the CPU backend
    ggml_context * ctx_far = ggml_init(ip);
    ggml_tensor * w_far[K];
    for (int k = 0; k < K; k++) {
        w_far[k] = ggml_new_tensor_2d(ctx_far, GGML_TYPE_F32, NE, NE);
    }
    ggml_tensor * w_wide = ggml_new_tensor_2d(ctx_far, GGML_TYPE_F32, 2*NE, 2*NE);
    ggml_backend_buffer_t buf_far = ggml_backend_alloc_ctx_tensors(ctx_far, cpu);
    ggml_backend_buffer_set_usage(buf_far, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int k = 0; k < K; k++) {
        fill_f32(w_far[k], 100 + k);
    }
    fill_f32(w_wide, 200);

    ggml_init_params gp = { ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(256, false), nullptr, true };
    ggml_context * ctx_g = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx_g, 256, false);

    // the shared tensor, produced on the dummy device
    ggml_tensor * shared = ggml_mul_mat(ctx_g, w_src, x); // [NE, NC]
    ggml_set_name(shared, "shared");

    ggml_tensor * out = nullptr;

    if (kind == CASE_RESHAPE) {
        // the production shape: a reshape of the tensor crosses first, then the tensor itself. The consumers need
        // different shapes, so the second copy tensor has to alias the first's memory rather than replace it.
        ggml_tensor * r  = ggml_reshape_2d(ctx_g, shared, 2*NE, NC/2);
        ggml_tensor * c1 = ggml_mul_mat(ctx_g, w_wide, r);                 // [2*NE, NC/2]
        ggml_build_forward_expand(gf, c1);

        ggml_tensor * b1 = ggml_mul_mat(ctx_g, w_near[0], ggml_reshape_2d(ctx_g, c1, NE, NC));
        ggml_build_forward_expand(gf, b1);

        ggml_tensor * c2 = ggml_mul_mat(ctx_g, w_far[0], shared);          // [NE, NC], same bytes as r
        ggml_build_forward_expand(gf, c2);

        ggml_tensor * b2 = ggml_mul_mat(ctx_g, w_near[1], c2);
        ggml_build_forward_expand(gf, b2);

        out = ggml_add(ctx_g, b1, b2);
    } else if (kind == CASE_MODIFIED) {
        // consumer 1 reads the whole tensor ...
        ggml_tensor * v1 = ggml_view_2d(ctx_g, shared, NE, NC, shared->nb[1], 0);
        ggml_tensor * c1 = ggml_mul_mat(ctx_g, w_far[0], v1);
        ggml_build_forward_expand(gf, c1);

        // ... then the base is written in place on the dummy device ...
        ggml_tensor * e   = ggml_mul_mat(ctx_g, w_mod, x);
        ggml_tensor * mod = ggml_add_inplace(ctx_g, shared, e);
        ggml_build_forward_expand(gf, mod);

        // ... and consumer 2 reads the same view again, which must see the new value
        ggml_tensor * v2 = ggml_view_2d(ctx_g, shared, NE, NC, shared->nb[1], 0);
        ggml_tensor * c2 = ggml_mul_mat(ctx_g, w_far[1], v2);
        ggml_build_forward_expand(gf, c2);

        out = ggml_add(ctx_g, c1, c2);
    } else {
        for (int k = 0; k < K; k++) {
            ggml_tensor * v = kind == CASE_SAME_VIEWS
                ? ggml_view_2d(ctx_g, shared, NE, NC, shared->nb[1], 0)
                : ggml_view_2d(ctx_g, shared, NE,  1, shared->nb[1], k*shared->nb[1]);

            // runs on the CPU backend: its weight is in a buffer the dummy device does not support
            ggml_tensor * far = ggml_mul_mat(ctx_g, w_far[k], v);
            ggml_build_forward_expand(gf, far);

            // back on the dummy device, so that the next consumer starts a new split there
            ggml_tensor * back = ggml_mul_mat(ctx_g, w_near[k], far);
            ggml_build_forward_expand(gf, back);

            out = out ? ggml_add(ctx_g, out, back) : back;
            ggml_build_forward_expand(gf, out);
        }
    }

    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    std::vector<ggml_backend_t> backends;
    if (dev) {
        backends.push_back(&dev->backend);
    }
    backends.push_back(cpu);

    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends.data(), nullptr, (int) backends.size(), 256, /* parallel */ false, /* op_offload */ false);

    const size_t reads_before = dev ? dev->n_reads : 0;
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_sched_synchronize(sched);
    if (n_reads) {
        *n_reads = dev ? dev->n_reads - reads_before : 0;
    }

    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buf_far);
    ggml_backend_buffer_free(buf_near);
    ggml_free(ctx_g);
    ggml_free(ctx_far);
    ggml_free(ctx_near);

    return result;
}

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    GGML_ASSERT(cpu);
    ggml_backend_cpu_set_n_threads(cpu, 1);

    dummy_dev * dev = dummy_dev_init();

    struct {
        case_kind    kind;
        const char * desc;
        size_t       expect_reads;
    } cases[] = {
        // K consumers of one identical view: the bytes must leave the device once
        { CASE_SAME_VIEWS, "same view x K",              1 },
        // K consumers of K different sub-views: nothing can be shared, each is copied
        { CASE_DISJOINT,   "disjoint sub-views x K",     K },
        // the base is written between the two consumers of an identical view: the copy must be redone
        { CASE_MODIFIED,   "same view, base modified",   2 },
        // a reshape of the tensor and the tensor itself: the same bytes in two shapes, moved once
        { CASE_RESHAPE,    "tensor + reshape of it",     1 },
    };

    int n_fail = 0;

    for (const auto & c : cases) {
        size_t reads = 0;
        std::vector<float> got = run_case(c.kind, dev, cpu, &reads);
        std::vector<float> ref = run_case(c.kind, nullptr, cpu, nullptr);

        const bool ok_val   = got.size() == ref.size() &&
                              memcmp(got.data(), ref.data(), got.size()*sizeof(float)) == 0;
        const bool ok_reads = reads == c.expect_reads;

        printf("  %-28s copies out of the device: %2zu (expected %2zu) %-5s  vs single backend: %s\n",
                c.desc, reads, c.expect_reads, ok_reads ? "OK" : "WRONG",
                ok_val ? "bit-identical" : "DIFFERENT");

        n_fail += (ok_val ? 0 : 1) + (ok_reads ? 0 : 1);
    }

    dummy_dev_free(dev);
    ggml_backend_free(cpu);

    printf("%s\n", n_fail == 0 ? "ALL OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
