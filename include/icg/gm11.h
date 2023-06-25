#ifndef ICG_GM11_H_
#define ICG_GM11_H_

#include<iostream>
#include"NumCpp.hpp"
// #include "NumCpp/NumCpp.hpp"
using namespace std;
 
class GM11
{
public:
    vector<vector<double>> data; // 已有数据
    int sampleNum; // 考虑的样本个数（考虑data的最后num个样本）
    int dimension;
    vector<double> predictPose;
public:
    GM11();
    GM11(vector<vector<double>> &data,int d):data(data), sampleNum(d)
    {
        dimension = data[0].size();
        predictPose.assign(dimension, 0);
    }

    void run();
    
    void examine(int index);

    void train(int index);

    void predict(int index, int k);

    double evaluate(nc::NdArray<double>&X0_hat, nc::NdArray<double>&X0);

    // vector<double> getPredictPose(){
    //     return predictPose;
    // }
};

template<class T>
void print(vector<T>& v){
    for(auto i: v){
        cout<<i<<" ";
    }
    cout<<endl;
}

vector<double> cumsum(vector<double>& v){
	vector<double> p;
	double sum=0;
    for (int j=0;j<v.size();j++)
    {
        sum+=v[j];
        p.push_back(sum);
    }
    return p;
}

void GM11::run(){
    for(int i=0; i<dimension; i++){
        examine(i);
        train(i);
    }
}

void GM11::examine(int index){
    
    vector<double> X0;
    for(int i=data.size()-sampleNum; i<data.size(); i++){
        X0.push_back(data[i][index]);
    }
    // cout<<"X0: "<<endl;
    // print(X0);
    vector<double> X1 = cumsum(X0);
    // cout<<"X1: "<<endl;
    // print(X1);
    vector<double> rho;
    for(int i=1; i<X0.size(); i++){
        rho.push_back(X0[i] / X1[i-1]);
    }
    // cout<<"rho: "<<endl;
    // print(rho);
    vector<double> rho_ratio;
    for(int i=0; i<rho.size()-1; i++){
        rho_ratio.push_back(rho[i + 1] / rho[i]); //越界了
    }

    // cout<<"rho_ratio: "<<endl;
    // print(rho_ratio);

    bool flag = true;
    for(int i=2; i<rho.size()-1; i++){
        if (rho[i] > 0.5 || rho[i + 1] / rho[i] >= 1)
            flag = false;
    }
    if (rho[-1] > 0.5)
        flag = false;

    // if (flag)
    //     cout<<"数据通过光滑校验"<<endl;
    // else
    //     cout<<"数据未通过光滑校验"<<endl;

    // '''判断是否通过级比检验'''
    vector<double> lambds;
    for(int i=1; i<X0.size(); i++){
        lambds.push_back(X0[i - 1] / X0[i]);
    }
    double X_min = nc::exp (-2.0 / (X0.size() + 1));
    double X_max = nc::exp (2.0 / (X0.size() + 1));
    for(auto lambd : lambds){
        // cout<<lambd<<" "<< X_min << " "<< X_max <<endl;
        if(lambd < X_min or lambd > X_max)
            // cout<<"该数据未通过级比检验"<<endl;
            return;
    }
    // cout<<"该数据通过级比检验"<<endl;
}

void GM11::train(int index){
    vector<double> X0;
    for(int i=data.size()-sampleNum; i<data.size(); i++){
        X0.push_back(data[i][index]);
    }

    vector<double> X1 = cumsum(X0);

    vector<double> vec;
    for(int k=1; k<X1.size(); k++){
        vec.push_back(-0.5 * (X1[k - 1] + X1[k]));
    }
    
    auto Z = nc::NdArray<double> (vec). reshape(X1.size() - 1, 1);
    vector<double> X0t{X0.begin()+1, X0.end()};
    nc::NdArray<double> X0_t{X0.begin()+1, X0.end()};
    nc::NdArray<double> X0_t2{-0.032198, -0.032198, -0.032198, -0.032198};

    auto A = nc::NdArray<double>(X0.begin()+1, X0.end()).reshape(Z.size(), 1);
    auto B = nc::hstack({Z,  nc::ones<double>(1, Z.size()).reshape(Z.size(), 1)});

    //求常微分方程中的a u参数
    auto vec_t = nc::linalg::inv(nc::matmul(B.transpose(), B)).dot(B.transpose()).dot(A);
    
    // cout<<vec_t[0]<<endl;
    // cout<<vec_t[1]<<endl;

    double a = vec_t[0];
    double u = vec_t[1];

    auto f = [&](double k){
        return (X0[0] - u / a) * nc::exp(-a * k) + u / a;
    };

    vector<double> X1_hat_v;
    for(int i=0; i<sampleNum+1; i++){
        X1_hat_v.push_back(f(i));
    }
    // cout<<"X1_hat_v"<<endl;
    // print(X1_hat_v);
    nc::NdArray<double> X1_hat(X1_hat_v);
    nc::NdArray<double> X0_hat = nc::diff(X1_hat);
    X0_hat = nc::hstack({nc::NdArray<double>({X1_hat[0]}), X0_hat});

    nc::NdArray<double> X0_temp(X0);
    // cout<<"X0_hat"<<endl;
    // X0_hat.print();
    // cout<<"X0_temp"<<endl;
    // X0_temp.print();
    // cout<<"预测值: "<<endl;
    // cout<<X0_hat[sampleNum]<<endl;
    // cout<<"size: "<<predictPose.size()<<endl;
    // cout<<"index: "<<index<<endl;
    predictPose[index] = X0_hat[sampleNum];
    nc::NdArray<double> X0_hat_temp(X0_hat.begin(), X0_hat.end()-1);
    double C = evaluate(X0_hat_temp, X0_temp); // 后验差比
    // 后验差比不在[0, 0.65]区间则不采用灰色预测值，取最后一个样本作为预测值
    if(!(C <= 0.65 && C >= 0)){
        predictPose[index] = X0.back();
    }
}

void GM11::predict(int index, int k){
    // vector<double> X1_hat;
    // for(int i=0; i<k; i++){
    //     X1_hat.push_back(i);
    // }
}

double GM11::evaluate(nc::NdArray<double>&X0_hat, nc::NdArray<double>&X0){
    auto S1 = nc::stdev(X0)[0];
    auto S2 = nc::stdev(X0 - X0_hat)[0];

    // cout<<"S1, S2: "<<S1<<" "<<S2<<endl;
    auto C = S2 / S1;
    auto Pe = nc::mean(X0 - X0_hat);

    // Pe.print();
    // X0.print();
    // X0_hat.print();

    auto temp = nc::abs((X0 - X0_hat - Pe[0])) < 0.6745 * S1;
    auto p = nc::count_nonzero(temp)[0] / X0.size();
    
    // cout<<"原数据样本标准差："<<S1<<endl;
    // cout<<"残差样本标准差: "<<S2<<endl;
    // cout<<"后验差比: "<<C<<endl;
    // cout<<"小误差概率p: "<<p<<endl;

    return C;
}

#endif