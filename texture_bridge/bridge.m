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

// Create a GPU-private r32Float/r16Float 3D texture and upload `data`
// (C-order [z][y][x], x fastest). Private storage gives Metal freedom to
// use the tiled texture layout preferred by the hardware sampler. Page-
// aligned, row-aligned inputs are wrapped without a CPU copy; other layouts
// use a bounded shared staging buffer and padded rows for the blit encoder.
// Returns a retained MTLTexture*.
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
        td.storageMode = MTLStorageModePrivate;
        id<MTLTexture> tex = [g_dev newTextureWithDescriptor:td];
        if (!tex) { set_err(@"texture allocation failed"); return NULL; }

        NSUInteger esz = half ? 2 : 4;
        NSUInteger srcRow = (NSUInteger)nx * esz;
        NSUInteger srcImage = srcRow * (NSUInteger)ny;
        NSUInteger srcBytes = srcImage * (NSUInteger)nz;
        NSUInteger page = (NSUInteger)getpagesize();

        // A buffer-to-texture blit requires 256-byte row alignment. For the
        // common power-of-two volumes numpy also supplies a page-aligned base,
        // so Metal can wrap the caller's storage for the synchronous upload.
        id<MTLBuffer> direct = nil;
        if (((uintptr_t)data % page) == 0 && (srcBytes % page) == 0 &&
            (srcRow % 256u) == 0) {
            @try {
                direct = [g_dev newBufferWithBytesNoCopy:(void *)data
                                                   length:srcBytes
                                                  options:MTLResourceStorageModeShared |
                                                          MTLResourceHazardTrackingModeUntracked
                                              deallocator:nil];
            } @catch (NSException *e) {
                direct = nil;
            }
        }

        if (direct) {
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit copyFromBuffer:direct
                    sourceOffset:0
               sourceBytesPerRow:srcRow
             sourceBytesPerImage:srcImage
                      sourceSize:MTLSizeMake(nx, ny, nz)
                       toTexture:tex
                destinationSlice:0
                destinationLevel:0
               destinationOrigin:MTLOriginMake(0, 0, 0)];
            [blit endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) {
                set_err([NSString stringWithFormat:@"texture upload failed: %@",
                                                   cb.error]);
                return NULL;
            }
        } else {
            // Bound fallback staging to 256 MiB. This matters for large or
            // oddly-sized volumes where a full padded copy could otherwise
            // double plan-construction memory.
            NSUInteger stageRow = (srcRow + 255u) & ~255u;
            NSUInteger stageImage = stageRow * (NSUInteger)ny;
            const NSUInteger maxStage = 256u * 1024u * 1024u;
            NSUInteger layers = MAX((NSUInteger)1, maxStage / stageImage);
            layers = MIN(layers, (NSUInteger)nz);
            id<MTLBuffer> staging =
                [g_dev newBufferWithLength:stageImage * layers
                                   options:MTLResourceStorageModeShared];
            if (!staging) {
                set_err(@"texture staging allocation failed");
                return NULL;
            }
            const unsigned char *src = (const unsigned char *)data;
            for (NSUInteger z0 = 0; z0 < (NSUInteger)nz; z0 += layers) {
                NSUInteger depth = MIN(layers, (NSUInteger)nz - z0);
                unsigned char *dst = (unsigned char *)staging.contents;
                if (stageRow == srcRow) {
                    memcpy(dst, src + z0 * srcImage, depth * srcImage);
                } else {
                    for (NSUInteger z = 0; z < depth; ++z) {
                        for (NSUInteger y = 0; y < (NSUInteger)ny; ++y) {
                            memcpy(dst + z * stageImage + y * stageRow,
                                   src + (z0 + z) * srcImage + y * srcRow,
                                   srcRow);
                        }
                    }
                }

                id<MTLCommandBuffer> cb = [g_queue commandBuffer];
                id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                [blit copyFromBuffer:staging
                        sourceOffset:0
                   sourceBytesPerRow:stageRow
                 sourceBytesPerImage:stageImage
                          sourceSize:MTLSizeMake(nx, ny, depth)
                           toTexture:tex
                    destinationSlice:0
                    destinationLevel:0
                   destinationOrigin:MTLOriginMake(0, 0, z0)];
                [blit endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.error) {
                    set_err([NSString stringWithFormat:
                        @"texture upload failed at z=%lu: %@",
                        (unsigned long)z0, cb.error]);
                    return NULL;
                }
            }
        }
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
