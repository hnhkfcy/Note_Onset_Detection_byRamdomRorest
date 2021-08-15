// test.cpp : 定义控制台应用程序的入口点。
//


#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream> 
#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace cv;

struct WavData {
public:
	int16_t* data;

	int channels;
	long size;
	int rate;
	WavData() {
		data = NULL;
		size = 0;
	}
};
#define step 4096
#define merge 4
int loadWavFile(const char* fname, WavData *ret) {

	/*if( (access(fname, 06 )) != 0 )
	{
		cout<<fname<<" not exise"<<endl;		
		return -1;
	}*/
	
	FILE* fp = fopen(fname, "rb");
	if (fp) {

		char id[5];
		int32_t size;
		int16_t format_tag, channels, block_align, bits_per_sample;
		int32_t format_length, sample_rate, avg_bytes_sec, data_size;

		fread(id, sizeof(char), 4, fp);
		id[4] = '\0';

		if (!strcmp(id, "RIFF")) {
			fread(&size, sizeof(int16_t), 2, fp);
			fread(id, sizeof(char), 4, fp);
			id[4] = '\0';

			char d;
			if (!strcmp(id, "WAVE")) {
				fread(id, sizeof(char), 4, fp);
				fread(&format_length, sizeof(int16_t), 2, fp);
				fread(&format_tag, sizeof(int16_t), 1, fp);
				fread(&channels, sizeof(int16_t), 1, fp);
				ret->channels = channels;
				fread(&sample_rate, sizeof(int16_t), 2, fp);
				ret->rate = sample_rate;
				fread(&avg_bytes_sec, sizeof(int16_t), 2, fp);
				fread(&block_align, sizeof(int16_t), 1, fp);
				fread(&bits_per_sample, sizeof(int16_t), 2, fp);
				
				while (1) {
					fread(&d, sizeof(char), 1, fp);
					if (d == 'd') {
						id[0] = d;
						fread(&d, sizeof(char), 1, fp);
						if (d == 'a') {
							id[1] = d;
							fread(&d, sizeof(char), 1, fp);
							if (d == 't') {
								id[2] = d;
								fread(&d, sizeof(char), 1, fp);
								if (d == 'a') {
									id[3] = d;
									id[4] = '\0';
									break;
								}
								else
									continue;
							}
							else
								continue;
						}
						else
							continue;

					}
				}
				//fread(id, sizeof(char), 4, fp);
				fread(&data_size, sizeof(int16_t), 2, fp);

				ret->size = data_size / sizeof(int16_t);
				// 动态分配了空间，记得要释放
	
				ret->data = (int16_t*)malloc(data_size);

				fread(ret->data, sizeof(int16_t), ret->size, fp);

			}
			else {
				cout << "Error: RIFF File but not a wave file\n";
			}
		}
		else {
			cout << "ERROR: not a RIFF file\n";
		}
	}
	fclose(fp);
	return 0;
}

void freeSource(WavData* data) {
	if(data->data)
	free(data->data);

}


int handle_line(char* dir,char* file_line,std::vector<int>& frames, WavData& data, string& filename) {

	std::vector<std::string> resVec;
	std::string strs = std::string(file_line);

	string flag = ",";
	size_t pos = strs.find(flag), startp = 0;
	size_t size = strs.size();

	while (pos != std::string::npos)
	{
		std::string x = strs.substr(0, pos);
		resVec.push_back(x);
		strs = strs.substr(pos + 1, size);
		pos = strs.find(flag);
	}
	resVec.push_back(strs);

	if (resVec.size() <= 2 || resVec.size()%2==0){

		return resVec.size();
	}
	char txt[600];
	filename = resVec[0].c_str();
	cout<<filename<<endl;
	sprintf(txt, dir, resVec[0].c_str());
	if(loadWavFile(txt, &data)==-1)
	{
		cout<<"loadWavFile error"<<endl;
		return -1;
	}
	frames.resize(resVec.size() - 1);

	for (auto i = 0; i < frames.size(); i++) {
		frames[i] = atof(resVec[i + 1].c_str()) * data.rate;
	}
	
	return frames.size();
}

Mat data2mat(WavData data,int& nrows) {
	Mat tdata = Mat(1, data.size, CV_16SC1, data.data), step_data;
	
	int ns = ((data.size) / step + 1)*step - data.size;
	cv::copyMakeBorder(tdata, tdata, 0, 0, 0, ns, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	nrows = tdata.cols / step;
	step_data = tdata.reshape(0, nrows);
	cv::Mat integer, data_full = cv::Mat::zeros(cv::Size(step / merge, nrows), CV_32FC1);
	for (int i = 0; i < nrows; i++) {
		cv::Mat arow = step_data.row(i);
		cv::resize(arow, arow, Size(arow.cols / merge, 1));
		cv::Scalar mean, std;
		cv::meanStdDev(arow, mean, std);
		if (std[0] != 0)
			arow.convertTo(data_full.row(i), CV_32FC1, 1 / std[0], -mean[0] / std[0]);
	}
	return data_full;
}
vector<int> randpermC(int N)
{   
    vector<int> r_vec;    
    int *arr = (int*)malloc(N*sizeof(int)); 
    int count = 0;
    memset(arr,0,N*sizeof(int));
    //srand(time(NULL));
    while(count<N)
    {
        int val = rand()%N;
        if (!arr[val])
        {
            //printf("%d ",val);
		r_vec.push_back(val);
            arr[val]=1;
            ++count;
        }
    }
    free(arr);
    arr = NULL;
	return r_vec;
}
int main()
{
//train data
#if 1
	ifstream inFile("/home/fan/rtree_bank/audio.txt", ios::in);

	Mat data256, all_labels;


	char* ss = new char[300];
	while (inFile.getline(ss, 300)) {

		std::vector<int> frames;
		WavData data;
		string wavefile;
		char* dir{ "/home/fan/rtree_bank/trainingset/training_set/%s" };
		
		int re=handle_line(dir,ss, frames, data, wavefile);
		
		if(re < 2 || re%2!=0){
			cout<<"error="<<re<<endl;
			continue;
		}
		int nrows;
		Mat data_full = data2mat(data,nrows);
		std::vector<float> label_data(nrows, 0);

		for (int j = 0; j < frames.size(); j = j + 2) {
			int id = frames[j] / step;
			int eid = (frames[j + 1] + 1) / step;

			for (int i = id; i < eid + 1; i++) {
				int end_id = min((i + 1)*step, frames[j + 1]);
				int start_id = max(i*step, frames[j]);
				if (end_id > start_id)
				{

					label_data[i] += (float)(end_id - start_id) / step;
				}

			}

		}

		cv::Mat label(nrows, 1, CV_32FC1, label_data.data());

		if (all_labels.empty()) {
			all_labels = label.clone();
			data256 = data_full.clone();
		}
		else {
			vconcat(all_labels, label, all_labels);
			vconcat(data256, data_full, data256);
		}

		freeSource(&data);


	}

	cout << "Now is Training\n Please wait a moment" << endl;


	printf("%d,%d    %d,%d\n",all_labels.cols,all_labels.rows,data256.cols,data256.rows);
	vector<int> r_v=randpermC(all_labels.rows);
	cv::Mat train_labels(all_labels.size(),all_labels.type());
	cv::Mat train_datas(data256.size(),data256.type());
	for(int i=0;i<all_labels.rows;i++){
		all_labels.row(r_v[i]).copyTo(train_labels.row(i));
		data256.row(r_v[i]).copyTo(train_datas.row(i));
	}
	Ptr<cv::ml::RTrees> m_trees;
	m_trees = cv::ml::RTrees::create();
	m_trees->setMaxDepth(10);
	m_trees->setMinSampleCount(10);
	m_trees->setRegressionAccuracy(0);
	m_trees->setUseSurrogates(false);
	m_trees->setMaxCategories(2);
	m_trees->setCalculateVarImportance(true);
	m_trees->setActiveVarCount(false); 
	m_trees->setTermCriteria({ cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,100, 0.01 });
	//return 0;	
	m_trees->train(train_datas, cv::ml::ROW_SAMPLE, train_labels);
	cout << "Trained\n saving model" << endl;

	//数据准备
	m_trees->save("../tree.bin");
#endif

// test data
	string xml{ "../tree.bin" };

	cv::Ptr<cv::ml::RTrees> dtrees = cv::ml::RTrees::load(xml);
	
	ifstream testFile;
	testFile.open("/home/fan/rtree_bank/trainingset/example.txt", ios::in);
	ofstream outFile("outs.txt");
	char* ts = new char[300];
	while (testFile.getline(ts, 300)) {


		std::vector<int> frames;
		WavData test_data;
		string wavefile;
		char* dir{ "/home/fan/rtree_bank/trainingset/training_set/%s" };
		handle_line(dir,ts, frames, test_data, wavefile);
		outFile << wavefile << "  ";
		cout << wavefile << "\n";
		int test_rows;
		Mat test_mat = data2mat(test_data, test_rows);
		cv::Mat res;
		vector<float> rr(test_mat.rows, 0);
		bool ent = true;
		bool first = true;
		bool allbig=true;
		vector<vector<float>> r_vectors;
		vector<vector<int>> n_vectors;
		vector<float> r_vector;
		vector<int > n_vector;

		for (int i = 0; i < test_mat.rows; i++) {
			float r = dtrees->predict(test_mat.row(i), res);
			if (r > 0.1) {
				ent = false;
				rr[i] = r;
				first = true;
				r_vector.push_back(r);
				n_vector.push_back(i);

			}else{
				ent = true;
				allbig=allbig | false;
			}

			if (ent&&first) {
				if (r_vector.size() > 0) {
					r_vectors.push_back(r_vector);
					n_vectors.push_back(n_vector);
					r_vector.clear();
					n_vector.clear();

				}
				first = false;
			}
		}
		if(allbig&&first){
			if (r_vector.size() > 0) {
				r_vectors.push_back(r_vector);
				n_vectors.push_back(n_vector);
				r_vector.clear();
				n_vector.clear();

			}
		}
		int nr = 0;
		float max = 0;
		int maxid;

		for (auto rs : r_vectors) {
			max = 0;
			int ss = rs.size()*step / test_data.rate;
			for (int i = 0; i < rs.size()- ss; i++) {
				float sum=0;
				for (int s = 0; s <= ss; s++) {
					sum += rs[i+s];
				}
				if (sum > max) {
					max = sum;
					maxid = n_vectors[nr][i];
				}
			}
			nr++;

			float tem = (maxid + 1 - rr[maxid])*step / test_data.rate;
			cout << " bank start " << tem << endl;
			outFile << tem << " ";
		}



		outFile << endl;
		continue;

	}

	testFile.close();


	return 0;
}

