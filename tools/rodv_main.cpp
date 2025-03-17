#include "rodv/variant_caller.h"

int main(int argc, char* argv[]) {
    try {
        rodv::RuntimeOptions options = ParseArguments(argc, argv);
        
        rodv::BamReader reader(options.bam_path);
        rodv::VariantCaller caller(options.model_path);
        
        auto variants = caller.CallVariants(reader);
        OutputVCF(variants, options.output_path);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}