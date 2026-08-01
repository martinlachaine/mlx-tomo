// mlx-tomo hardware-texture bridge.
//
// Minimal C-ABI dylib giving Python access to Metal 3D textures and the
// hardware trilinear sampler, which mx.fast.metal_kernel cannot bind.
// The projection kernel MSL is generated in Python (single source of truth
// for geometry) and compiled here at plan time.
//
// Interop contract (same pattern as mlx-nufft's vkfft_bridge):
// - Output buffers are caller-owned host pointers (page-padded numpy
//   arrays); page-aligned ones are wrapped zero-copy with
//   newBufferWithBytesNoCopy and a nil deallocator, others fall back to an
//   internal staging buffer + memcpy. (MLX arrays cannot be used: MLX 0.32
//   dlpack exports Metal GPU addresses, and its data pointers are heap
//   sub-allocations.)
// - All dispatches are synchronous (waitUntilCompleted) so MLX never races
//   with the external command queue.
//
// Build: texture_bridge/build.sh

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string.h>

static id<MTLDevice> g_dev = nil;
static id<MTLCommandQueue> g_queue = nil;
static char g_err[4096] = "";

static void set_err(NSString *msg) {
    strncpy(g_err, msg.UTF8String, sizeof(g_err) - 1);
    g_err[sizeof(g_err) - 1] = 0;
}

const char *mcb_error(void) { return g_err; }

int mcb_init(void) {
    if (g_dev) return 0;
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) { set_err(@"no Metal device"); return -1; }
    g_queue = [g_dev newCommandQueue];
    if (!g_queue) { set_err(@"newCommandQueue failed"); g_dev = nil; return -2; }
    return 0;
}

// Compile `src` and return a retained MTLComputePipelineState* for `fn`.
void *mcb_compile(const char *src, const char *fn) {
    if (mcb_init() != 0) return NULL;
    @autoreleasepool {
        NSError *err = nil;
        MTLCompileOptions *opts = [MTLCompileOptions new];
        id<MTLLibrary> lib =
            [g_dev newLibraryWithSource:[NSString stringWithUTF8String:src]
                                options:opts
                                  error:&err];
        if (!lib) {
            set_err([NSString stringWithFormat:@"compile failed: %@", err]);
            return NULL;
        }
        id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn]];
        if (!f) { set_err(@"kernel function not found"); return NULL; }
        id<MTLComputePipelineState> pso =
            [g_dev newComputePipelineStateWithFunction:f error:&err];
        if (!pso) {
            set_err([NSString stringWithFormat:@"pipeline failed: %@", err]);
            return NULL;
        }
        return (void *)CFBridgingRetain(pso);
    }
}

// Create an r32Float/r16Float 3D texture and upload `data` (C-order
// [z][y][x], x fastest). Returns a retained MTLTexture*.
void *mcb_texture3d(const void *data, int nx, int ny, int nz, int half) {
    if (mcb_init() != 0) return NULL;
    if (!data || nx < 1 || ny < 1 || nz < 1) {
        set_err(@"invalid texture dims/data");
        return NULL;
    }
    @autoreleasepool { @try {
        MTLTextureDescriptor *td = [MTLTextureDescriptor new];
        td.textureType = MTLTextureType3D;
        td.pixelFormat = half ? MTLPixelFormatR16Float : MTLPixelFormatR32Float;
        td.width = (NSUInteger)nx;
        td.height = (NSUInteger)ny;
        td.depth = (NSUInteger)nz;
        td.usage = MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModeShared;
        id<MTLTexture> tex = [g_dev newTextureWithDescriptor:td];
        if (!tex) { set_err(@"texture allocation failed"); return NULL; }
        NSUInteger esz = half ? 2 : 4;
        MTLRegion region = MTLRegionMake3D(0, 0, 0, nx, ny, nz);
        [tex replaceRegion:region
               mipmapLevel:0
                     slice:0
                 withBytes:data
               bytesPerRow:(NSUInteger)nx * esz
             bytesPerImage:(NSUInteger)nx * (NSUInteger)ny * esz];
        return (void *)CFBridgingRetain(tex);
    } @catch (NSException *e) {
        set_err([NSString stringWithFormat:@"texture creation threw: %@", e]);
        return NULL;
    } }
}

// Dispatch: grid (nu, nv, nviews), threadgroup (tgx, tgy, 1).
// out_ptr must remain valid for the duration of the (synchronous) call.
int mcb_project(void *pso_h, void *tex_h,
                const float *views, int nviews,
                void *out_ptr, unsigned long long out_bytes,
                int nu, int nv, int tgx, int tgy) {
    if (mcb_init() != 0) return -1;
    if (!pso_h || !tex_h) { set_err(@"NULL pipeline/texture handle"); return -6; }
    if (!views || !out_ptr || nviews < 1) { set_err(@"NULL/empty inputs"); return -7; }
    @autoreleasepool { @try {
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pso_h;
        id<MTLTexture> tex = (__bridge id<MTLTexture>)tex_h;
        if (tgx < 1 || tgy < 1 ||
            (NSUInteger)tgx * (NSUInteger)tgy > pso.maxTotalThreadsPerThreadgroup) {
            set_err([NSString stringWithFormat:
                @"threadgroup %dx%d exceeds pipeline max %lu",
                tgx, tgy, (unsigned long)pso.maxTotalThreadsPerThreadgroup]);
            return -5;
        }

        id<MTLBuffer> vbuf = [g_dev newBufferWithBytes:views
                                                length:(NSUInteger)nviews * 16 * 4
                                               options:MTLResourceStorageModeShared];
        if (!vbuf) { set_err(@"views buffer failed"); return -2; }

        // Zero-copy wrap of the caller's host pointer is only legal when
        // it is page-aligned and the length a page multiple; Metal throws
        // (not nil) otherwise. Fall back to staging.
        id<MTLBuffer> obuf = nil;
        int staged = 1;
        size_t page = (size_t)getpagesize();
        if (((uintptr_t)out_ptr % page) == 0 && (out_bytes % page) == 0) {
            @try {
                obuf = [g_dev newBufferWithBytesNoCopy:out_ptr
                                                length:(NSUInteger)out_bytes
                                               options:MTLResourceStorageModeShared |
                                                       MTLResourceHazardTrackingModeUntracked
                                           deallocator:nil];
                staged = (obuf == nil);
            } @catch (NSException *e) {
                obuf = nil;
                staged = 1;
            }
        }
        if (staged) {
            obuf = [g_dev newBufferWithLength:(NSUInteger)out_bytes
                                      options:MTLResourceStorageModeShared];
            if (!obuf) { set_err(@"output buffer failed"); return -3; }
        }

        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setTexture:tex atIndex:0];
        [enc setBuffer:vbuf offset:0 atIndex:0];
        [enc setBuffer:obuf offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(nu, nv, nviews)
            threadsPerThreadgroup:MTLSizeMake(tgx, tgy, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            set_err([NSString stringWithFormat:@"dispatch failed: %@", cb.error]);
            return -4;
        }
        if (staged)
            memcpy(out_ptr, obuf.contents, (size_t)out_bytes);
        return 0;
    } @catch (NSException *e) {
        set_err([NSString stringWithFormat:@"dispatch threw: %@", e]);
        return -8;
    } }
}

void mcb_release(void *obj) {
    if (obj) CFBridgingRelease(obj);
}
