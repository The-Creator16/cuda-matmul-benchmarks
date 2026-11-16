#include "host.hpp"
#include "kernels.cuh"
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

#define CUDA(call) do { cudaError_t e = (call); if(e != cudaSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e)); } while(0)
#define BLAS(call) do { cublasStatus_t e = (call); if(e != CUBLAS_STATUS_SUCCESS) \
    throw std::runtime_error(std::string(#call) + ": cuBLAS status " + std::to_string(e)); } while(0)

struct Buffer {
    float* ptr = nullptr;
    explicit Buffer(std::size_t count) { CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr), count*sizeof(float))); }
    ~Buffer() { if(ptr) cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct Stream {
    cudaStream_t value;
    Stream() { CUDA(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
struct Handle {
    cublasHandle_t value;
    Handle() { BLAS(cublasCreate(&value)); }
    ~Handle() { cublasDestroy(value); }
};
struct Event {
    cudaEvent_t value;
    Event() { CUDA(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
};

struct Options {
    bench::Shape shape{512, 512, 512};
    int samples = 11, repeats = 20, warmup = 5, seed = 42;
    std::string kernel = "all", pattern = "random", csv;
    bool validate_only = false, profile = false;
};

void usage() {
    std::cout << "matmul_bench [--m M --n N --k K] [--kernel all|naive|tiled16|tiled32|cublas]\n"
                 "  [--samples 11 --repeats 20 --warmup 5 --seed 42] [--csv PATH]\n"
                 "  [--pattern random|zero|ones|identity|alternating] [--validate-only]\n"
                 "  [--profile]  Warm up then launch one selected operation in a profiler range.\n"
                 "All numeric values: 1..16384. All output matrices use FP32.\n";
}

Options parse(int argc, char** argv) {
    Options o;
    for(int i=1; i<argc; i=i+1) {
        std::string key = argv[i];
        if(key == "--help") { usage(); std::exit(0); }
        if(key == "--validate-only") { o.validate_only = true; continue; }
        if(key == "--profile") { o.profile = true; continue; }
        if(i+1 >= argc) throw std::runtime_error("Missing value for " + key);
        std::string value = argv[++i];
        if(key == "--kernel") o.kernel = value;
        else if(key == "--pattern") o.pattern = value;
        else if(key == "--csv") o.csv = value;
        else if(key == "--m") o.shape.m = bench::positive_int(value);
        else if(key == "--n") o.shape.n = bench::positive_int(value);
        else if(key == "--k") o.shape.k = bench::positive_int(value);
        else if(key == "--samples") o.samples = bench::positive_int(value);
        else if(key == "--repeats") o.repeats = bench::positive_int(value);
        else if(key == "--warmup") o.warmup = bench::positive_int(value);
        else if(key == "--seed") o.seed = bench::positive_int(value);
        else throw std::runtime_error("Unknown option: " + key);
    }
    const std::vector<std::string> names{"all", "naive", "tiled16", "tiled32", "cublas"};
    if(std::find(names.begin(), names.end(), o.kernel) == names.end()) {
        throw std::runtime_error("Unknown kernel: " + o.kernel);
    }
    if(o.profile && (o.kernel == "all" || o.validate_only || !o.csv.empty())) {
        throw std::runtime_error("--profile requires one kernel and no --csv/--validate-only");
    }
    return o;
}

void launch(const std::string& name, const Buffer& a, const Buffer& b, Buffer& c,
            bench::Shape s, cudaStream_t stream, cublasHandle_t handle) {
    if(name == "cublas") {
        const float alpha = 1.0f, beta = 0.0f;
        // Row-major C=A*B equals column-major C^T=B^T*A^T, without a copy.
        // B is column-major [N,K], A is [K,M], C is [N,M].
        BLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             s.n, s.m, s.k, &alpha,
             b.ptr, CUDA_R_32F, s.n, a.ptr, CUDA_R_32F, s.k,
             &beta, c.ptr, CUDA_R_32F, s.n,
             CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
    } else {
        int tile = name == "tiled32" ? 32 : 16;
        dim3 block(tile, tile), grid((s.n+tile-1)/tile, (s.m+tile-1)/tile);
        if(name == "naive") {
            matmul_naive<<<grid, block, 0, stream>>>(a.ptr, b.ptr, c.ptr, s.m, s.n, s.k);
        } else if(name == "tiled16") {
            matmul_tiled<16><<<grid, block, 0, stream>>>(a.ptr, b.ptr, c.ptr, s.m, s.n, s.k);
        } else {
            matmul_tiled<32><<<grid, block, 0, stream>>>(a.ptr, b.ptr, c.ptr, s.m, s.n, s.k);
        }
        CUDA(cudaGetLastError());
    }
}

int main(int argc, char** argv) {
    try {
        Options o = parse(argc, argv);
        auto s = o.shape;
        CUDA(cudaSetDevice(0)); // Select a physical device via CUDA_VISIBLE_DEVICES.
        cudaDeviceProp prop{};
        CUDA(cudaGetDeviceProperties(&prop, 0));
        if((o.kernel == "all" || o.kernel == "tiled32") && prop.maxThreadsPerBlock < 1024) {
            throw std::runtime_error("tiled32 requires 1024 threads per block; select a different kernel");
        }
        int runtime = 0, driver = 0;
        CUDA(cudaRuntimeGetVersion(&runtime));
        CUDA(cudaDriverGetVersion(&driver));
        Stream stream;
        Handle handle;
        BLAS(cublasSetStream(handle.value, stream.value));
        BLAS(cublasSetPointerMode(handle.value, CUBLAS_POINTER_MODE_HOST));
        int blas_version = 0;
        BLAS(cublasGetVersion(handle.value, &blas_version));
        std::cerr << "GPU=" << prop.name << " CC=" << prop.major << '.' << prop.minor
                  << " runtime=" << runtime << " driver=" << driver << " cuBLAS=" << blas_version
                  << " precision=FP32_PEDANTIC TF32=off\n";

        std::vector<std::string> names = o.kernel == "all"
            ? std::vector<std::string>{"naive", "tiled16", "tiled32", "cublas"}
            : std::vector<std::string>{o.kernel};
        std::size_t ac = bench::elements(s.m, s.k), bc = bench::elements(s.k, s.n);
        std::size_t cc = bench::elements(s.m, s.n), free_bytes = 0, total_bytes = 0;
        CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
        if((ac+bc+cc)*sizeof(float) > free_bytes*0.85) {
            throw std::runtime_error("Insufficient free GPU memory with 15% workspace headroom");
        }
        std::vector<float> ha(ac), hb(bc), output(cc), reference;
        bench::fill_inputs(ha, hb, s, o.seed, o.pattern);
        Buffer a(ac), b(bc), c(cc);
        CUDA(cudaMemcpyAsync(a.ptr, ha.data(), ac*sizeof(float), cudaMemcpyHostToDevice, stream.value));
        CUDA(cudaMemcpyAsync(b.ptr, hb.data(), bc*sizeof(float), cudaMemcpyHostToDevice, stream.value));
        CUDA(cudaStreamSynchronize(stream.value));
        auto run = [&](const std::string& name) { launch(name, a, b, c, s, stream.value, handle.value); };

        if(o.profile) {
            for(int w=0; w<o.warmup; w=w+1) run(o.kernel);
            CUDA(cudaStreamSynchronize(stream.value));
            CUDA(cudaProfilerStart());
            run(o.kernel);
            CUDA(cudaStreamSynchronize(stream.value));
            CUDA(cudaProfilerStop());
            std::cerr << "Profile launch completed; this mode does not validate or benchmark.\n";
            return 0;
        }

        // Independent CPU double reference: exhaustive for small cases, sampled for large ones.
        auto indices = bench::cpu_indices(s, o.seed);
        std::vector<double> cpu;
        for(auto i : indices) cpu.push_back(bench::reference_at(ha, hb, s, i));
        auto read_output = [&]() {
            CUDA(cudaMemcpyAsync(output.data(), c.ptr, cc*sizeof(float), cudaMemcpyDeviceToHost, stream.value));
            CUDA(cudaStreamSynchronize(stream.value));
        };
        auto poison = [&]() {
            // NaN bytes expose unwritten output cells (all variants must overwrite C).
            CUDA(cudaMemsetAsync(c.ptr, 0xff, cc*sizeof(float), stream.value));
        };
        poison();
        run("cublas");
        read_output();
        reference = output;
        bench::Error ref_error;
        for(std::size_t j=0; j<indices.size(); j=j+1) ref_error.add(reference[indices[j]], cpu[j]);
        for(float x : reference) {
            if(!std::isfinite(x)) throw std::runtime_error("Non-finite cuBLAS reference");
        }
        if(ref_error.failed) throw std::runtime_error("cuBLAS failed the independent CPU reference");

        std::vector<bench::Error> errors(names.size()), cpu_errors(names.size());
        for(std::size_t v=0; v<names.size(); v=v+1) {
            poison();
            run(names[v]);
            read_output();
            for(std::size_t i=0; i<cc; i=i+1) errors[v].add(output[i], reference[i]);
            for(std::size_t j=0; j<indices.size(); j=j+1) cpu_errors[v].add(output[indices[j]], cpu[j]);
            std::cerr << names[v] << ": full_cublas_check=" << errors[v].checked
                      << " cpu_check=" << cpu_errors[v].checked << " max_abs=" << errors[v].max_abs
                      << " cpu_max_abs=" << cpu_errors[v].max_abs << '\n';
            if(errors[v].failed || cpu_errors[v].failed) {
                throw std::runtime_error("Validation failed for " + names[v]);
            }
        }
        if(o.validate_only) { std::cerr << "PASS\n"; return 0; }

        // A separate initial warmup for every variant, followed by randomized rounds.
        for(const auto& name : names) {
            for(int w=0; w<o.warmup; w=w+1) run(name);
        }
        CUDA(cudaStreamSynchronize(stream.value));
        Event start, stop;
        std::vector<std::vector<float>> times(names.size());
        std::vector<std::size_t> order;
        for(std::size_t v=0; v<names.size(); v=v+1) order.push_back(v);
        std::mt19937 rng(o.seed);
        for(int sample=0; sample<o.samples; sample=sample+1) {
            std::shuffle(order.begin(), order.end(), rng);
            for(auto v : order) {
                CUDA(cudaEventRecord(start.value, stream.value));
                for(int r=0; r<o.repeats; r=r+1) run(names[v]);
                CUDA(cudaEventRecord(stop.value, stream.value));
                CUDA(cudaEventSynchronize(stop.value));
                float elapsed = 0.0f;
                CUDA(cudaEventElapsedTime(&elapsed, start.value, stop.value));
                if(!(elapsed > 0.0f) || !std::isfinite(elapsed)) {
                    throw std::runtime_error("Invalid timing; increase repeats");
                }
                times[v].push_back(elapsed/o.repeats);
            }
        }
        // Recheck the final timed result (the last shuffled variant).
        read_output();
        bench::Error after;
        for(std::size_t i=0; i<cc; i=i+1) after.add(output[i], reference[i]);
        if(after.failed) throw std::runtime_error("Post-timing validation failed");

        std::ofstream file;
        if(!o.csv.empty()) {
            file.open(o.csv);
            if(!file) throw std::runtime_error("Cannot write CSV: " + o.csv);
        }
        std::ostream& out = o.csv.empty() ? std::cout : file;
        out << "m,n,k,variant,pattern,seed,sample,repeats,ms,gflops,max_abs_vs_cublas,max_scaled_vs_cublas,cpu_checked,cpu_max_abs,validated\n";
        out << std::setprecision(10);
        for(std::size_t v=0; v<names.size(); v=v+1) {
            for(int t=0; t<o.samples; t=t+1) {
                double ms = times[v][t];
                out << s.m << ',' << s.n << ',' << s.k << ',' << names[v] << ',' << o.pattern
                    << ',' << o.seed << ',' << t << ',' << o.repeats << ',' << ms << ','
                    << bench::flops(s)/(ms*1e6) << ',' << errors[v].max_abs << ','
                    << errors[v].max_scaled << ',' << cpu_errors[v].checked << ','
                    << cpu_errors[v].max_abs << ",1\n";
            }
        }
        out.flush();
        if(!out) throw std::runtime_error("Failed to flush CSV output");
        return 0;
    } catch(const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
