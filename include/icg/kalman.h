#ifndef ICG_KALMAN_H_
#define ICG_KALMAN_H_

#include<iostream>
#include<Eigen/Dense>
using namespace std;
 
class Kalman
{
public:
    int m_StateSize; //state variable's dimenssion
    int m_MeaSize; //measurement variable's dimession
    int m_USize; //control variables's dimenssion
    Eigen::VectorXd m_x;  //状态量
    Eigen::VectorXd m_u;  //输入矩阵
    Eigen::VectorXd m_z;  //观测量
    Eigen::MatrixXd m_A;  //状态转移矩阵
    Eigen::MatrixXd m_B;  //控制矩阵
    Eigen::MatrixXd m_P;  //先验估计协方差
    Eigen::MatrixXd m_H;  //观测矩阵
    Eigen::MatrixXd m_R;  //测量噪声协方差
    Eigen::MatrixXd m_Q;  //过程噪声协方差
    Eigen::MatrixXd m_iden_mat;
public:
    Kalman()
    {
        cout<<"Kalman construct function......"<<endl;
    }
    Kalman(int statesize,int measize,int usize):m_StateSize(statesize),m_MeaSize(measize),m_USize(usize)
    {
        if(m_StateSize==0&&m_MeaSize==0)
        {
            cout<<"Init........."<<endl;
        }
    
        m_x.resize(statesize);
        m_x.setZero();

        m_z.resize(measize);
        m_z.setZero();

        // 影响预测值，如果预测值为外部输入值则用不上
        m_u.resize(usize);
        m_u.setZero();
        m_B.resize(statesize,usize);
        m_B.setZero();
    
        m_A.resize(statesize,statesize);
        m_A.setIdentity();
    
        m_P.resize(statesize,statesize);
        m_P.setIdentity();
    
        m_H.resize(measize,statesize);
        m_H.setZero();
    
        m_R.resize(measize,measize);
        m_R.setZero();
    
        m_Q.resize(statesize,statesize);
        m_Q.setZero();
    
        m_iden_mat.resize(statesize,statesize);
        m_iden_mat.setIdentity();
    }
    // void Init(Eigen::Matrix<double,6,1> &x,Eigen::Matrix<double,6,6> &P,Eigen::Matrix2d &R,Eigen::Matrix<double,6,6> &Q);
    // Eigen::VectorXd predict(Eigen::Matrix<double,6,6> &A);
    // Eigen::VectorXd predict(Eigen::Matrix<double,6,6> &A,Eigen::MatrixXd &B,Eigen::VectorXd &u);
    // void Update(Eigen::Matrix<double,2,6> &H,Eigen::Matrix<double,2,1> z_meas);
 
    void Init_Par(Eigen::VectorXd& x,Eigen::MatrixXd& P,Eigen::MatrixXd& R,Eigen::MatrixXd& Q,Eigen::MatrixXd& A,Eigen::MatrixXd& B,Eigen::MatrixXd& H,Eigen::VectorXd& u)
    {
        m_x=x;
        m_P=P;
        m_R=R;
        m_Q=Q;
        m_A=A;
        m_B=B;
        m_H=H;
        m_u=u;

        // cout<<"m_x状态量: "<<endl;
        // cout<<m_x.matrix()<<endl;
        // cout<<"m_u输入矩阵: "<<endl;
        // cout<<m_u.matrix()<<endl;
        // cout<<"观测量: "<<endl;
        // cout<<m_z.matrix()<<endl;
        // cout<<"m_A状态转移矩阵: "<<endl;
        // cout<<m_A.matrix()<<endl;
        // cout<<"m_B控制矩阵: "<<endl;
        // cout<<m_B.matrix()<<endl;
        // cout<<"m_P先验估计协方差: "<<endl;
        // cout<<m_P.matrix()<<endl;
        // cout<<"m_H观测矩阵: "<<endl;
        // cout<<m_H.matrix()<<endl;
        // cout<<"m_R测量噪声协方差: "<<endl;
        // cout<<m_R.matrix()<<endl;
        // cout<<"m_Q过程噪声协方差: "<<endl;
        // cout<<m_Q.matrix()<<endl;
    }

    void Predict_State()
    {
        Eigen::VectorXd tmp_state=m_A*m_x+m_B*m_u;
        m_x=tmp_state;
        // cout<<"m_x预测后状态量: "<<endl;
        // cout<<m_x<<endl;
    }

    // 用输入的矩阵来作为预测值
    void Predict_State2(Eigen::VectorXd& predicted_x)
    {
        // m_x=predicted_x;
        // cout<<"m_x预测后状态量: "<<endl;
        // cout<<m_x<<endl;
    }

    void Predict_Cov()
    {
        Eigen::MatrixXd tmp_cov=m_A*m_P*m_A.transpose()+m_Q;
        m_P=tmp_cov;
    }

    Eigen::VectorXd Mea_Resd(Eigen::VectorXd& z)
    {
        m_z=z;
        Eigen::VectorXd tmp_res=m_z-m_H*m_x;
        return tmp_res;
    }

    // 卡尔曼增益
    Eigen::MatrixXd Cal_Gain()
    {
        Eigen::MatrixXd tmp_gain=m_P*m_H.transpose()*(m_H*m_P*m_H.transpose()+m_R).inverse();
        return tmp_gain;
    }

    // 更新状态量，用卡尔曼增益与测量值
    void Update_State()
    {
        Eigen::MatrixXd kal_gain=Cal_Gain();
        Eigen::VectorXd mea_res=Mea_Resd(m_z);
        // cout<<"m_z观测量: "<<endl;
        // cout<<m_z<<endl;
        // cout<<"卡尔曼增益: "<<endl;
        // cout<<kal_gain<<endl;
        // cout<<"mea_res: "<<endl;
        // cout<<mea_res<<endl;
        m_x=m_x+kal_gain*mea_res;
    }

    void Update_Cov()
    {
        Eigen::MatrixXd kal_gain=Cal_Gain();
        Eigen::MatrixXd tmp_mat=kal_gain*m_H;
        m_P=(m_iden_mat-tmp_mat)*m_P;
    }

    // 输入预测量与观测量来做卡尔曼滤波
    void Run(Eigen::VectorXd x, Eigen::VectorXd z0, float deltaPercent){
        // if(deltaPercent == 0)
        //     m_R.setIdentity();
        // else{
        //     m_R.setIdentity();
        //     m_R *= 0.2;
        // }
        // m_R.setIdentity();
        // m_R *= deltaPercent<0.2?0.2:deltaPercent;
        // cout<<m_R<<endl;
        // cout<<"******卡尔曼滤波开始*******"<<endl;
        // cout<<"滤波前m_x状态量: "<<endl;
        // cout<<m_x<<endl;
        // Predict_State();
        Predict_State2(x);
        Predict_Cov();
        Mea_Resd(z0);
        Cal_Gain();
        Update_State();
        Update_Cov();
        // cout<<"滤波后m_x状态量: "<<endl;
        // cout<<m_x<<endl;
        // cout<<"m_A状态转移矩阵: "<<endl;
        // cout<<m_A<<endl;
        // cout<<"m_B控制矩阵: "<<endl;
        // cout<<m_B<<endl;
        // cout<<"m_H观测矩阵: "<<endl;
        // cout<<m_H<<endl;
        // cout<<"m_P先验估计协方差: "<<endl;
        // cout<<m_P.matrix()<<endl;
        // cout<<"m_H观测矩阵: "<<endl;
        // cout<<m_H.matrix()<<endl;
        // cout<<"m_R测量噪声协方差: "<<endl;
        // cout<<m_R.matrix()<<endl;
        // cout<<"m_Q过程噪声协方差: "<<endl;
        // cout<<m_Q.matrix()<<endl;
        // cout<<"******卡尔曼滤波结束*******"<<endl;
    }

    Eigen::VectorXd Result(){
        return m_x;
    }
};

// void Kalman::Init_Par(Eigen::VectorXd& x,Eigen::MatrixXd& P,Eigen::MatrixXd& R,Eigen::MatrixXd& Q,Eigen::MatrixXd& A,Eigen::MatrixXd& B,Eigen::MatrixXd& H,Eigen::VectorXd& u)
// {
//     m_x=x;
//     m_P=P;
//     m_R=R;
//     m_Q=Q;
//     m_A=A;
//     m_B=B;
//     m_H=H;
//     m_u=u;

//     // cout<<"m_x状态量: "<<endl;
//     // cout<<m_x.matrix()<<endl;
//     cout<<"m_u输入矩阵: "<<endl;
//     cout<<m_u.matrix()<<endl;
//     // cout<<"观测量: "<<endl;
//     // cout<<m_z.matrix()<<endl;
//     // cout<<"m_A状态转移矩阵: "<<endl;
//     // cout<<m_A.matrix()<<endl;
//     cout<<"m_B控制矩阵: "<<endl;
//     cout<<m_B.matrix()<<endl;
//     // cout<<"m_P先验估计协方差: "<<endl;
//     // cout<<m_P.matrix()<<endl;
//     // cout<<"m_H观测矩阵: "<<endl;
//     // cout<<m_H.matrix()<<endl;
//     // cout<<"m_R测量噪声协方差: "<<endl;
//     // cout<<m_R.matrix()<<endl;
//     // cout<<"m_Q过程噪声协方差: "<<endl;
//     // cout<<m_Q.matrix()<<endl;
//     // Eigen::VectorXd m_x;  //状态量
//     // Eigen::VectorXd m_u;  //输入矩阵
//     // Eigen::VectorXd m_z;  //观测量
//     // Eigen::MatrixXd m_A;  //状态转移矩阵
//     // Eigen::MatrixXd m_B;  //控制矩阵
//     // Eigen::MatrixXd m_P;  //先验估计协方差
//     // Eigen::MatrixXd m_H;  //观测矩阵
//     // Eigen::MatrixXd m_R;  //测量噪声协方差
//     // Eigen::MatrixXd m_Q;  //过程噪声协方差
// }
 
// void Kalman::Predict_State()
// {
//     Eigen::VectorXd tmp_state=m_A*m_x+m_B*m_u;

//     cout<<"m_x预测前状态量: "<<endl;
//     cout<<m_x<<endl;
//     m_x=tmp_state;
//     cout<<"m_x预测后状态量: "<<endl;
//     cout<<m_x<<endl;
// }
 
// void Kalman::Predict_Cov()
// {
//     Eigen::MatrixXd tmp_cov=m_A*m_P*m_A.transpose()+m_Q;
//     m_P=tmp_cov;
// }
 
// Eigen::VectorXd Kalman::Mea_Resd(Eigen::VectorXd& z)
// {
//     m_z=z;
//     Eigen::VectorXd tmp_res=m_z-m_H*m_x;
//     return tmp_res;
// }
 
// Eigen::MatrixXd Kalman::Cal_Gain()
// {
//     Eigen::MatrixXd tmp_gain=m_P*m_H.transpose()*(m_H*m_P*m_H.transpose()+m_R).inverse();
//     return tmp_gain;
// }
 
// void Kalman::Update_State()
// {
//     Eigen::MatrixXd kal_gain=Cal_Gain();
//     Eigen::VectorXd mea_res=Mea_Resd(m_z);
//     cout<<"m_z观测量: "<<endl;
//     cout<<m_z.matrix()<<endl;
//     cout<<"m_x更新前状态量: "<<endl;
//     cout<<m_x.matrix()<<endl;
//     m_x=m_x+kal_gain*mea_res;
//     cout<<"m_x更新后状态量: "<<endl;
//     cout<<m_x.matrix()<<endl;
// }
 
// void Kalman::Update_Cov()
// {
//     Eigen::MatrixXd kal_gain=Cal_Gain();
//     Eigen::MatrixXd tmp_mat=kal_gain*m_H;
//     m_P=(m_iden_mat-tmp_mat)*m_P;
// }

// void Kalman::Run(double measure){
//     Eigen::VectorXd z0;
//     z0.resize(1);
//     z0(0)=measure;
//     cout<<"the "<<(1)<<" th time predict"<<endl;
//     Predict_State();
//     Predict_Cov();
//     Mea_Resd(z0);
//     Cal_Gain();
//     Update_State();
//     Update_Cov();
//     cout<<"m_x状态量: "<<endl;
//     cout<<m_x<<endl;
//     // cout<<"m_A状态转移矩阵: "<<endl;
//     // cout<<ka.m_A<<endl;
//     // cout<<"m_B控制矩阵: "<<endl;
//     // cout<<ka.m_B<<endl;
//     // cout<<"m_H观测矩阵: "<<endl;
//     // cout<<ka.m_H<<endl;
//     cout<<"*************"<<endl;
// }

#endif