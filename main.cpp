/*!
 * @author    gyu-don
 * @date      2019
 * @copyright BSD-3-Clause
 */
/*!
 * Original code was developed by
 * @author    PEZY Computing, K.K.
 * @date      2019
 * @copyright BSD-3-Clause
 */

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

enum Gate {
    H = 1,
    X,
    Z,
    CX,
    CZ,
    PHASE,
    M,
    M_VAL,
    P0,
};

struct Params {
    int gate;
    int target;
    int ctrl;
    int sampl;
    double lambda;
    double phi;
    double theta;
    Params(int g, int t, int c = -1) : gate(g), target(t), ctrl(c) {}
    Params(int g, int t, double theta, double phi = 0., double lambda = 0.) : gate(g), target(t), theta(theta), phi(phi), lambda(lambda) {}
};

namespace {
    std::mt19937 mt(0);
    inline void  initVector(std::vector<double>& src_re, std::vector<double>& src_im)
    {
        for (auto& s : src_re) {
            s = 0;
        }
        for (auto& s : src_im) {
            s = 0;
        }
        src_re[0] = 1;
    }

    void cpuAdd(size_t num, std::vector<double>& dst, const std::vector<double>& src0, const std::vector<double>& src1)
    {
        for (size_t i = 0; i < num; ++i) {
            dst[i] = src0[i] + src1[i];
        }
    }

    inline size_t getFileSize(std::ifstream& file)
    {
        file.seekg(0, std::ios::end);
        size_t ret = file.tellg();
        file.seekg(0, std::ios::beg);

        return ret;
    }

    inline void loadFile(std::ifstream& file, std::vector<char>& d, size_t size)
    {
        d.resize(size);
        file.read(reinterpret_cast<char*>(d.data()), size);
    }

    cl::Program createProgram(cl::Context& context, const std::vector<cl::Device>& devices, const std::string& filename)
    {
        std::ifstream file;
        file.open(filename, std::ios::in | std::ios::binary);

        if (file.fail()) {
            throw "can not open kernel file";
        }

        size_t            filesize = getFileSize(file);
        std::vector<char> binary_data;
        loadFile(file, binary_data, filesize);

        cl::Program::Binaries binaries;
        binaries.push_back(std::make_pair(&binary_data[0], filesize));

        return cl::Program(context, devices, binaries, nullptr, nullptr);
    }

    cl::Program createProgram(cl::Context& context, const cl::Device& device, const std::string& filename)
    {
        std::vector<cl::Device> devices { device };
        return createProgram(context, devices, filename);
    }

    std::vector<int> pzcRun(int n_qubits, std::vector<Params>& ops, std::vector<double>* host_re = nullptr, std::vector<double>* host_im = nullptr)
    {
        auto measured = std::vector<int>(n_qubits);
        size_t num = 1 << n_qubits;
        try {
            // Get Platform
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            const auto& Platform = platforms[0];

            // Get devices
            std::vector<cl::Device> devices;
            Platform.getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);

            // Use first device.
            const auto& device = devices[0];

            // Create Context.
            auto context = cl::Context(device);

            // Create CommandQueue.
            auto command_queue = cl::CommandQueue(context, device, 0);

            // Create Program.
            // Load compiled binary file and create cl::Program object.
            auto program = createProgram(context, device, "kernel/kernel.pz");

            // Get workitem size.
            // sc1-64: 8192  (1024 PEs * 8 threads)
            // sc2   : 15782 (1984 PEs * 8 threads)
            size_t global_work_size = 0;
            {
                std::string device_name;
                device.getInfo(CL_DEVICE_NAME, &device_name);

                size_t global_work_size_[3] = { 0 };
                device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &global_work_size_);

                global_work_size = global_work_size_[0];
                if (device_name.find("PEZY-SC2") != std::string::npos) {
                    global_work_size = std::min(global_work_size, (size_t)15872);
                }

                std::cout << "Use device : " << device_name << std::endl;
                std::cout << "workitem   : " << global_work_size << std::endl;
            }

            // Create Buffers.
            auto device_vec_re = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * num);
            auto device_vec_im = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * num);
            auto device_p0 = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double));

            // Send src.
            // command_queue.enqueueWriteBuffer(device_vec_re, true, 0, sizeof(double) * num, &vec_re[0]);
            // cl::Event write_event;
            // command_queue.enqueueWriteBuffer(device_vec_im, true, 0, sizeof(double) * num, &vec_im[0], 0, &write_event);
            // write_event.wait();
            // std::cerr << "write_event.wait() done" << std::endl;
            auto kernel = cl::Kernel(program, "initvec");
            kernel.setArg(0, num);
            kernel.setArg(1, device_vec_re);
            kernel.setArg(2, device_vec_im);
            command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);

            for (auto &op : ops) {
                switch (op.gate) {
                    case H:
                    {
                        auto kernel = cl::Kernel(program, "hgate");
                        kernel.setArg(0, num);
                        kernel.setArg(1, (uint64_t)1 << op.target);
                        kernel.setArg(2, device_vec_re);
                        kernel.setArg(3, device_vec_im);
                        command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);
                        break;
                    }
                    case X:
                    {
                        auto kernel = cl::Kernel(program, "xgate");
                        kernel.setArg(0, num);
                        kernel.setArg(1, (uint64_t)1 << op.target);
                        kernel.setArg(2, device_vec_re);
                        kernel.setArg(3, device_vec_im);
                        command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);
                        break;
                    }
                    case Z:
                    {
                        auto kernel = cl::Kernel(program, "zgate");
                        kernel.setArg(0, num);
                        kernel.setArg(1, (uint64_t)1 << op.target);
                        kernel.setArg(2, device_vec_re);
                        kernel.setArg(3, device_vec_im);
                        command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);
                        break;
                    }
                    case CX:
                    {
                        auto kernel = cl::Kernel(program, "cxgate");
                        kernel.setArg(0, num);
                        kernel.setArg(1, (uint64_t)1 << op.target);
                        kernel.setArg(2, (uint64_t)1 << op.ctrl);
                        kernel.setArg(3, device_vec_re);
                        kernel.setArg(4, device_vec_im);
                        command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);
                        break;
                    }
                    case CZ:
                    {
                        auto kernel = cl::Kernel(program, "czgate");
                        kernel.setArg(0, num);
                        kernel.setArg(1, (uint64_t)1 << op.target);
                        kernel.setArg(2, (uint64_t)1 << op.ctrl);
                        kernel.setArg(3, device_vec_re);
                        kernel.setArg(4, device_vec_im);
                        command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);
                        break;
                    }
                    case P0:
                    case M:
                    case M_VAL:
                    {
                        auto kernel = cl::Kernel(program, "p0_base8");
                        kernel.setArg(0, num);
                        kernel.setArg(1, (uint64_t)1 << op.target);
                        kernel.setArg(2, device_vec_re);
                        kernel.setArg(3, device_vec_im);
                        kernel.setArg(4, device_p0);
                        command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);
                        command_queue.enqueueReadBuffer(device_p0, true, 0, sizeof(double), &op.phi);
                        if (op.gate == P0) break;
                        if (op.gate == M) {
                            std::uniform_real_distribution<> rnd01(0.0, 1.0);
                            op.theta = rnd01(mt);
                        }
                        if (op.phi < op.theta) {
                            auto kernel = cl::Kernel(program, "collapse_to_1");
                            measured[op.target] = 1;
                            kernel.setArg(0, num);
                            kernel.setArg(1, (uint64_t)1 << op.target);
                            kernel.setArg(2, device_vec_re);
                            kernel.setArg(3, device_vec_im);
                            kernel.setArg(4, op.phi);
                            command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);
                        }
                        else {
                            auto kernel = cl::Kernel(program, "collapse_to_0");
                            measured[op.target] = 0;
                            kernel.setArg(0, num);
                            kernel.setArg(1, (uint64_t)1 << op.target);
                            kernel.setArg(2, device_vec_re);
                            kernel.setArg(3, device_vec_im);
                            kernel.setArg(4, op.phi);
                            command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NullRange);
                        }
                        break;
                    }
                }
            }
            // Get dst.
            if (host_re) {
                host_re->resize(num);
                command_queue.enqueueReadBuffer(device_vec_re, true, 0, sizeof(double) * num, host_re->data());
            }
            if (host_im) {
                host_im->resize(num);
                command_queue.enqueueReadBuffer(device_vec_im, true, 0, sizeof(double) * num, host_im->data());
            }

            // Finish all commands.
            command_queue.flush();
            std::cerr << "command_queue.flush() done" << std::endl;
            command_queue.finish();
            std::cerr << "command_queue.finish() done" << std::endl;
        } catch (const cl::Error& e) {
            std::stringstream msg;
            msg << "CL Error : " << e.what() << " " << e.err();
            throw std::runtime_error(msg.str());
        }
        return measured;
    }

    /*
       bool chk(const std::vector<double>& re, const std::vector<double>& im)
       {
       assert(actual.size() == expected.size());

       bool   is_true     = true;
       size_t error_count = 0;

       const size_t num = actual.size();
       for (size_t i = 0; i < num; ++i) {
       if (fabs(actual[i] - expected[i]) > 1.e-7) {

       if (error_count < 10) {
       std::cerr << "# ERROR " << i << " " << actual[i] << " " << expected[i] << std::endl;
       }
       error_count++;
       is_true = false;
       }
       }

       return is_true;
       }
       */

    bool verify(const std::vector<double>& actual, const std::vector<double>& expected)
    {
        assert(actual.size() == expected.size());

        bool   is_true     = true;
        size_t error_count = 0;

        const size_t num = actual.size();
        for (size_t i = 0; i < num; ++i) {
            if (fabs(actual[i] - expected[i]) > 1.e-7) {

                if (error_count < 10) {
                    std::cerr << "# ERROR " << i << " " << actual[i] << " " << expected[i] << std::endl;
                }
                error_count++;
                is_true = false;
            }
        }

        return is_true;
    }
}

int main(int argc, char** argv)
{
    size_t n_qubits = 30;

    if (argc > 1) {
        n_qubits = strtol(argv[1], nullptr, 10);
    }

    std::cout << "n_qubits " << n_qubits << std::endl;

    std::vector<Params> ops;
    for (int i = 0; i < n_qubits; i+=2) {
        ops.push_back(Params(H, i));
        ops.push_back(Params(CX, i, i + 1));
    }
    for (int i = 0; i < n_qubits; i++) {
        ops.push_back(Params(M, i));
    }

    std::vector<double> re, im;
    bool dump_statevec = n_qubits < 5;
    auto vec = pzcRun(n_qubits, ops, dump_statevec ? &re : nullptr, dump_statevec ? &im : nullptr);
    std::cout << "pzcRun: done" << std::endl;
    if (dump_statevec) {
        for (int i = 0; i < (1 << n_qubits); i++) {
            std::cout << i << "\t";
            std::cout << re[i] << "\t";
            std::cout << im[i] << std::endl;
        }
    }
    for (auto v : vec) {
        std::cout << v;
    }
    std::cout << std::endl;

    /*
    for(size_t i=0; i < num; i++) {
        std::cout << i << "\t(" << vec_re[i] << " + i" << vec_im[i] << ")" << std::endl;
    }
    */
    // verify
    /*
       if (chk(vec_re, vec_im)) {
       std::cout << "PASS" << std::endl;
       } else {
       std::cout << "FAIL" << std::endl;
       }
       */

    return 0;
}
