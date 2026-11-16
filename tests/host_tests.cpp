#include "host.hpp"
#include <iostream>
#include <limits>

void check(bool ok, const char* message) {
    if(!ok) throw std::runtime_error(message);
}
int main() {
    try {
        // Non-square, hand-computed reference catches indexing/layout mistakes.
        bench::Shape s{2, 2, 3};
        std::vector<float> a{1,2,3,4,5,6}, b{7,8,9,10,11,12};
        std::vector<double> expected{58,64,139,154};
        for(std::size_t i=0; i<4; i=i+1) check(bench::reference_at(a,b,s,i)==expected[i], "CPU reference");
        bench::Error e;
        e.add(0.0005, 0.0);
        e.add(100.005, 100.0);
        check(e.failed==0, "absolute/relative tolerance");
        e.add(1.0, 0.0);
        e.add(std::numeric_limits<double>::quiet_NaN(), 0.0);
        e.add(0.0, std::numeric_limits<double>::infinity());
        check(e.failed==3, "bad results and non-finite values must fail");
        for(const auto& bad : {"0", "-1", "1.5", "12x", "16385", ""}) {
            bool rejected = false;
            try { bench::positive_int(bad); } catch(const std::exception&) { rejected = true; }
            check(rejected, "invalid CLI dimension accepted");
        }
        check(bench::positive_int("16384")==16384, "upper dimension boundary");
        check(bench::flops({2,3,4})==48, "FLOP count");
        auto idx = bench::cpu_indices({1024,1024,1024},42);
        check(idx.size()==256 && idx.front()==0 && idx.back()==1024*1024-1, "sampling coverage");
        check(idx==bench::cpu_indices({1024,1024,1024},42), "deterministic sample");
        std::vector<float> x(6), y(6), x2(6), y2(6);
        bench::fill_inputs(x,y,s,42,"random");
        bench::fill_inputs(x2,y2,s,42,"random");
        check(x==x2 && y==y2, "deterministic data");
        bench::fill_inputs(x,y,s,42,"ones");
        check(bench::reference_at(x,y,s,0)==3, "ones pattern");
        bench::fill_inputs(x,y,s,42,"alternating");
        check(bench::reference_at(x,y,s,0)==1, "cancellation pattern");
        std::cout << "Host tests PASS\n";
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
