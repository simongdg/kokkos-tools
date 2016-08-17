
#ifndef _H_KOKKOSP_KERNEL_INFO
#define _H_KOKKOSP_KERNEL_INFO

#include <stdio.h>
#include <sys/time.h>
#include <cstring>



double seconds() {
	struct timeval now;
	gettimeofday(&now, NULL);

	return (double) (now.tv_sec + (now.tv_usec * 1.0e-6));
}

enum KernelExecutionType {
	PARALLEL_FOR = 0,
	PARALLEL_REDUCE = 1,
	PARALLEL_SCAN = 2
};

class KernelPerformanceInfo {
	public:
		KernelPerformanceInfo(std::string kName, KernelExecutionType kernelType, uint32_t parID = 0) :
			kType(kernelType), current_team_size(NULL), current_vector_length(NULL), done(true), parent(false), pID(parID) {
			
			kernelName = (char*) malloc(sizeof(char) * (kName.size() + 1));
			strcpy(kernelName, kName.c_str());

			callCount = 0;
			time = 0;
		}
		
		KernelPerformanceInfo(std::string kName, KernelExecutionType kernelType, uint32_t* team_size, uint32_t* vector_length, uint32_t kID, uint32_t max_iteration_count, uint32_t nnz_per_row = 32) :
                        kType(kernelType), best_time(1.e6), maxVectLength(32)/*depends on ARCH*/, maxHWcapasity(1024)/*depends on ARCH*/, done(false), parent(true), uniqID(kID) {

                        kernelName = (char*) malloc(sizeof(char) * (kName.size() + 1));
                        strcpy(kernelName, kName.c_str());

                        callCount = 0;
                        time = 0;

			maxIter = max_iteration_count;

			best_vector_length = maxVectLength;
        		current_team_size = team_size;
        		current_vector_length = vector_length;
        		non_zeros = nnz_per_row;
        		*current_vector_length = 2;
        		
			while(*current_vector_length <non_zeros) *current_vector_length*=2;
        		if(*current_vector_length > maxVectLength) *current_vector_length = maxVectLength;
        		*current_team_size =  32; //maxHWcapasity / *current_vector_length;

                }

		~KernelPerformanceInfo() {
			free(kernelName);
		}

		KernelExecutionType getKernelType() {
			return kType;
		}

		void incrementCount() {
			callCount++;
		}

		void addTime(double t) {
			time   += t;
			timeSq += (t*t);
		
			if(callCount > maxIter)
				done = true;

			if(!done){
          			if(t<best_time) { best_time = t; best_vector_length = *current_vector_length; best_team_size = *current_team_size;}
          			*current_vector_length /= 2;
          			if(*current_vector_length <= non_zeros/8 && *current_vector_length < 1)
          			{
            				*current_team_size *= 2;
            				*current_vector_length = maxHWcapasity / *current_team_size;
					if(*current_vector_length > maxVectLength) *current_vector_length = maxVectLength;
					while(*current_vector_length > non_zeros) *current_vector_length/=2;
            				if(*current_team_size >= maxHWcapasity){
              					done = true;
              					*current_vector_length = best_vector_length;
              					*current_team_size = best_team_size;
            				}
          			}
			}
		}

		void addFromTimer() {
			addTime(seconds() - startTime);

			incrementCount();
		}

		void startTimer() {
			startTime = seconds();
		}

		uint64_t getCallCount() {
			return callCount;
		}

		double getTime() {
			return time;
		}

		double getTimeSq() {
			return timeSq;
		}

		char* getName() {
			return kernelName;
		}

		bool isParent(){
			return parent;
		}

		uint32_t getID(){
			return uniqID;
		}
	
		uint32_t parentID(){
			return pID;
		}
		
		void addCallCount(const uint64_t newCalls) {
			callCount += newCalls;
		}
		
		bool readFromFile(FILE* input) {
			uint32_t recordLen = 0;
			uint32_t actual_read = fread(&recordLen, sizeof(recordLen), 1, input);
	                if(actual_read != 1) return false;

			char* entry = (char*) malloc(recordLen);
                        fread(entry, recordLen, 1, input);

			uint32_t nextIndex = 0;
			uint32_t kernelNameLength;
			copy((char*) &kernelNameLength, &entry[nextIndex], sizeof(kernelNameLength));
			nextIndex += sizeof(kernelNameLength);
			
			if(strlen(kernelName) > 0) {
				free(kernelName);
			}

			kernelName = (char*) malloc( sizeof(char) * (kernelNameLength + 1));
			copy(kernelName, &entry[nextIndex], kernelNameLength);
			kernelName[kernelNameLength] = '\0';
			nextIndex += kernelNameLength;
			
			copy((char*) &callCount, &entry[nextIndex], sizeof(callCount));
			nextIndex += sizeof(callCount);
			
			copy((char*) &time, &entry[nextIndex], sizeof(time));
			nextIndex += sizeof(time);
			
			copy((char*) &timeSq, &entry[nextIndex], sizeof(timeSq));
			nextIndex += sizeof(timeSq);
			
			uint32_t kernelT = 0;
			copy((char*) &kernelT, &entry[nextIndex], sizeof(kernelT));
			nextIndex += sizeof(kernelT);
			
			if(kernelT == 0) {
				kType = PARALLEL_FOR;
			} else if(kernelT == 1) {
				kType = PARALLEL_REDUCE;
			} else if(kernelT == 2) {
				kType = PARALLEL_SCAN;
			}

			copy((char*) &parent, &entry[nextIndex], sizeof(parent));
                        nextIndex += sizeof(parent);
			
			free(entry);
                        return true;
		}
		
		void writeToFile(FILE* output) {
			const uint32_t kernelNameLen = (uint32_t) strlen(kernelName);
			
			const uint32_t recordLen = 
				sizeof(uint32_t) + 
				sizeof(char) * kernelNameLen +
				sizeof(uint64_t) +
				sizeof(double) +
				sizeof(double) + 
				sizeof(uint32_t) +
				sizeof(bool);
		
			uint32_t nextIndex = 0;
			char* entry = (char*) malloc(recordLen);

			copy(&entry[nextIndex], (char*) &kernelNameLen, sizeof(kernelNameLen));
			nextIndex += sizeof(kernelNameLen);
			
			copy(&entry[nextIndex], kernelName, kernelNameLen);
			nextIndex += kernelNameLen;
			
			copy(&entry[nextIndex], (char*) &callCount, sizeof(callCount));
			nextIndex += sizeof(callCount);
			
			copy(&entry[nextIndex], (char*) &time, sizeof(time));
			nextIndex += sizeof(time);
			
			copy(&entry[nextIndex], (char*) &timeSq, sizeof(timeSq));
			nextIndex += sizeof(timeSq);
			
			uint32_t kernelTypeOutput = (uint32_t) kType;
			copy(&entry[nextIndex], (char*) &kernelTypeOutput, sizeof(kernelTypeOutput));
			nextIndex += sizeof(kernelTypeOutput);

			copy(&entry[nextIndex], (char*) &parent, sizeof(parent));
                        nextIndex += sizeof(parent);
			
			fwrite(&recordLen, sizeof(uint32_t), 1, output);
			fwrite(entry, recordLen, 1, output);
			free(entry);
		}

	private:
		void copy(char* dest, const char* src, uint32_t len) {
			for(uint32_t i = 0; i < len; i++) {
				dest[i] = src[i];
			}
		}
	
		char* kernelName;
		uint64_t callCount;
		double time;
		double timeSq;
		double startTime;
		KernelExecutionType kType;
		double best_time;
		uint32_t best_vector_length;
    		uint32_t best_team_size;
    		uint32_t maxVectLength;
    		uint32_t maxHWcapasity;
    		uint32_t non_zeros;
		uint32_t uniqID;
		uint32_t pID;
		uint32_t maxIter;
    		uint32_t* current_team_size;
    		uint32_t* current_vector_length;
    		bool done;
		bool parent;
};

#endif
