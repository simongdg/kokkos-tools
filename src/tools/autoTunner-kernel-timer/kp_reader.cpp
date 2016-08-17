
#include <stdio.h>
#include <cinttypes>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <map>

#include "kp_kernel_info.h"

bool compareKernelPerformanceInfo(KernelPerformanceInfo* left, KernelPerformanceInfo* right) {
	return left->getTime() > right->getTime();
};

int find_index(std::vector<KernelPerformanceInfo*>& kernels,
	const char* kernelName) {
	
	for(int i = 0; i < kernels.size(); i++) {
		KernelPerformanceInfo* nextKernel = kernels[i];
	
		if(strcmp(nextKernel->getName(), kernelName) == 0) {
			return i;
		}	
	}
	
	return -1;
}

int main(int argc, char* argv[]) {

	if(argc == 1) {
		fprintf(stderr, "Did you specify any data files on the command line!\n");
		fprintf(stderr, "Usage: ./reader file1.dat [fileX.dat]*\n");
		exit(-1);
	}

        char delimiter   = ' ';
        int fixed_width  = 1;

        int commandline_args = 1;
        while( (commandline_args<argc ) && (argv[commandline_args][0]=='-') ) {
          if(strcmp(argv[commandline_args],"--delimiter")==0) {
            delimiter=argv[++commandline_args][0];
          }
          if(strcmp(argv[commandline_args],"--fixed-width")==0) {
            fixed_width=atoi(argv[++commandline_args]);
          }

          commandline_args++;
        }

	std::vector<KernelPerformanceInfo*> kernelInfo;
	double totalKernelsTime = 0;
	double totalExecuteTime = 0;
	uint64_t totalKernelsCalls = 0;

	for(int i = commandline_args; i < argc; i++) {
		FILE* the_file = fopen(argv[i], "rb");
		
		double fileExecuteTime = 0;
		fread(&fileExecuteTime, sizeof(fileExecuteTime), 1, the_file);
		
		totalExecuteTime += fileExecuteTime;

		while(! feof(the_file)) {
			KernelPerformanceInfo* new_kernel = new KernelPerformanceInfo("", PARALLEL_FOR);
			if(new_kernel->readFromFile(the_file)) {
			   if(strlen(new_kernel->getName()) > 0) {
				int kernelIndex = find_index(kernelInfo, new_kernel->getName());

				if(kernelIndex > -1) {
					kernelInfo[kernelIndex]->addTime(new_kernel->getTime());
					kernelInfo[kernelIndex]->addCallCount(new_kernel->getCallCount());
				} else {
					kernelInfo.push_back(new_kernel);
				}
                           }
			}
		}

		fclose(the_file);
	}

	std::sort(kernelInfo.begin(), kernelInfo.end(), compareKernelPerformanceInfo);

	bool parent;
	for(int i = 0; i < kernelInfo.size(); i++) {
		
		parent = kernelInfo[i]->isParent();
		if(!parent){
			totalKernelsTime += kernelInfo[i]->getTime();
			totalKernelsCalls += kernelInfo[i]->getCallCount();
		}
	}

	for(int i = 0; i < kernelInfo.size(); i++) {
		if(kernelInfo[i]->isParent())
		{
			const double callCountDouble = (double) kernelInfo[i]->getCallCount();
			if(fixed_width)
	                printf("%100s%c%30.5f%c%20" PRIu64 "%c%15.5f%c%7.3f%c%7.3f\n", kernelInfo[i]->getName(),
                        	delimiter,kernelInfo[i]->getTime(),
                        	delimiter,kernelInfo[i]->getCallCount(),
                        	delimiter,kernelInfo[i]->getTime() / callCountDouble,
                        	delimiter,(kernelInfo[i]->getTime() / totalKernelsTime) * 100.0,
                        	delimiter,(kernelInfo[i]->getTime() / totalExecuteTime) * 100.0 );
        	        else
                	printf("%s%c%f%c%" PRIu64 "%c%f%c%f%c%f\n", kernelInfo[i]->getName(),
                        	delimiter,kernelInfo[i]->getTime(),
                        	delimiter,kernelInfo[i]->getCallCount(),
                        	delimiter,kernelInfo[i]->getTime() / callCountDouble,
                        	delimiter,(kernelInfo[i]->getTime() / totalKernelsTime) * 100.0,
                        	delimiter,(kernelInfo[i]->getTime() / totalExecuteTime) * 100.0 );

			
			for(int j = 0; j < kernelInfo.size(); j++) {
				if(!(kernelInfo[j]->isParent()))
				{
					if(kernelInfo[i]->getID() == kernelInfo[j]->parentID())
					{
						const double callCountDouble2 = (double) kernelInfo[i]->getCallCount();
						if(fixed_width)
				                printf("%115s%c%15.5f%c%20" PRIu64 "%c%15.5f%c%7.3f%c%7.3f\n", kernelInfo[j]->getName(),
                        				delimiter,kernelInfo[j]->getTime(),
                        				delimiter,kernelInfo[j]->getCallCount(),
                        				delimiter,kernelInfo[j]->getTime() / callCountDouble2,
                        				delimiter,(kernelInfo[j]->getTime() / totalKernelsTime) * 100.0,
                        				delimiter,(kernelInfo[j]->getTime() / totalExecuteTime) * 100.0 );
                				else
                				printf("%s%c%f%c%" PRIu64 "%c%f%c%f%c%f\n", kernelInfo[j]->getName(),
                        				delimiter,kernelInfo[j]->getTime(),
                        				delimiter,kernelInfo[j]->getCallCount(),
                        				delimiter,kernelInfo[j]->getTime() / callCountDouble2,
                        				delimiter,(kernelInfo[j]->getTime() / totalKernelsTime) * 100.0,
                        				delimiter,(kernelInfo[j]->getTime() / totalExecuteTime) * 100.0 );
					}
				}	
			}
		}
	}

	printf("\n");
	printf("-------------------------------------------------------------------------\n");
	printf("Summary:\n");
	printf("\n");
	printf("Total Execution Time (incl. Kokkos + Non-Kokkos:       %20.5f seconds\n", totalExecuteTime);
	printf("Total Time in Kokkos kernels:                          %20.5f seconds\n", totalKernelsTime);
	printf("   -> Time outside Kokkos kernels:                     %20.5f seconds\n",
		(totalExecuteTime - totalKernelsTime));
	printf("   -> Percentage in Kokkos kernels:                    %20.2f %%\n",
		(totalKernelsTime / totalExecuteTime) * 100);
	printf("Total Calls to Kokkos Kernels:                         %20" PRIu64 "\n", totalKernelsCalls);
	printf("\n");
	printf("-------------------------------------------------------------------------\n");

	return 0;

}
