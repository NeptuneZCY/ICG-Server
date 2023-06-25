// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Manuel Stoiber, German Aerospace Center (DLR)

#ifndef ICG_INCLUDE_ICG_SERVER_H_
#define ICG_INCLUDE_ICG_SERVER_H_
// #include<icg/f.h>
#include<thread>
#include<stdio.h>
#include<iostream>
#include<cstring>
#include<string>
#include<stdlib.h>
#include<sys/fcntl.h>
#include<sys/socket.h>
#include<unistd.h>
#include<netinet/in.h>
#include<errno.h>
#include<sys/types.h>
#include<arpa/inet.h>
#include<icg/body.h>
#include<queue>
#define PORT 6000
// std::vector<int> poseVec = std::vector<int>();

namespace icg {
    class Server{
    private:
        static Server *server;
        static bool isConnect;
        static char receive_buffer[1024];
        static char send_buffer[1024];
        static int listenfd;
        static int clintfd;
        static int iret;
        Server();
    public:
        static std::queue<std::string> msgQueue;
        static Server *GetServer(){
            if(server == NULL){
                server = new Server();
            }

            return server;
        }
        static void Connect(){
            std::cout<< "Server Starting..." << std::endl;
            // int listenfd;
            listenfd=socket(AF_INET,SOCK_STREAM,0);// in socket code,it must be AF_INET(protocol) 
            if(listenfd==-1)
            {
                std::cout<< "socket create fail" << std::endl;
                return;
            }
            //second step bind ->server's ip&port for communication to socket created in fist step
            struct sockaddr_in serveraddr;
            memset(&serveraddr,0,sizeof(serveraddr));
            serveraddr.sin_family=AF_INET;
            //INADDR_ANY就是指定地址为0.0.0.0的地址，这个地址事实上表示不确定地址，或“所有地址”、“任意地址”。
            serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
            serveraddr.sin_port=htons(PORT);//specify port

            //端口占用
            int opt = 1;
            setsockopt(listenfd,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof( opt ));

            if(bind(listenfd,(struct sockaddr *)&serveraddr,sizeof(serveraddr))!=0)
            {
                std::cout<< "bind failed" << std::endl; 
                return;
            }
            //Third step ->Set socket to listening mode
            /*
            The listen function changes the active connection socket interface into the connected socket interface, 
            so that a process can accept the requests of other processes and become a server process. 
            In TCP server programming, the listen function changes the process into a server and specifies that the corresponding socket becomes a passive connection.
            */

            if(listen(listenfd,5)!=0)
            {
                std::cout<< "Listen failed" << std::endl;
                close(listenfd);
                return;
            }

            // 4th step -> receive client's request
            // int clintfd;//socket for client
            int socklen=sizeof(struct sockaddr_in);
            struct sockaddr_in client_addr;
            clintfd=accept(listenfd,(struct sockaddr*)&client_addr,(socklen_t *)&socklen);
            if(clintfd==-1)
                std::cout<< "connect failed" << std::endl;
            else{
                std::cout<< "client " << inet_ntoa(client_addr.sin_addr) << " has connnected" << std::endl;
                isConnect = true;
            }

            // 5th step ->connect with client,receive data and reply OK
            
            
            std::thread th1(receive);
            th1.detach();

            // std::thread th2(sendm);
            // th2.detach();

            // while(isConnect){

            // }
            // 6th close socket
            //close(listenfd); close(clintfd);
        }
        
        static void receive(){
            std::cout << "开始接收信息" << std::endl;
            // msgQueue = std::queue<std::string>();
            while (isConnect)
            {
                // int iret;
                memset(receive_buffer,0,sizeof(receive_buffer));
                iret=recv(clintfd,receive_buffer,sizeof(receive_buffer),0);
                if (iret<=0) 
                {
                printf("iret=%d\n",iret); break;  
                }
                printf("receive :%s\n",receive_buffer);
                std::string msg(receive_buffer);
                // "Next": ,"Track": ,"Reset":
                msgQueue.push(msg);
            }
            close(listenfd); close(clintfd);isConnect = false;

            Connect();
        }

        static void sendm(){
            while (isConnect)
            {
                memset(send_buffer,0,sizeof(send_buffer));
                strcpy(send_buffer,"ok");//reply cilent with "ok"
                if ( (iret=send(clintfd,send_buffer,strlen(send_buffer),0))<=0) 
                { 
                    perror("send");  
                    break; 
                }
                printf("send :%s\n",send_buffer);
            }
        }

        static void sendPose(std::shared_ptr<icg::Body> body_ptr){
            if (isConnect)
            {
                std::string msg;
                msg += body_ptr->name();
                auto mat = body_ptr->body2world_pose().matrix();
                for(int i=0; i<mat.rows(); i++){
                    for(int j=0; j<mat.cols(); j++){
                        msg += ',';
                        msg += std::to_string((float)mat(i, j));
                    }
                }

                msg += '@';

                memset(send_buffer,0,sizeof(send_buffer));
                strcpy(send_buffer,msg.c_str());//reply cilent with "ok"
                if ( (iret=send(clintfd,send_buffer,strlen(send_buffer),0))<=0) 
                { 
                    perror("send failed!"); 
                    return;
                }
                // printf("send :%s\n",send_buffer);
            }
        }

        static void sendMeanPose(std::string name, Transform3fA& pose){
            if (isConnect)
            {
                std::string msg;
                msg += name;
                auto mat = pose.matrix();
                for(int i=0; i<mat.rows(); i++){
                    for(int j=0; j<mat.cols(); j++){
                        msg += ',';
                        msg += std::to_string((float)mat(i, j));
                    }
                }

                msg += '@';

                memset(send_buffer,0,sizeof(send_buffer));
                strcpy(send_buffer,msg.c_str());//reply cilent with "ok"
                if ( (iret=send(clintfd,send_buffer,strlen(send_buffer),0))<=0) 
                { 
                    perror("send failed!"); 
                    return;
                }
                // printf("send :%s\n",send_buffer);
            }
        }

    };

    // bool Server::isConnect = false;
    // char Server::receive_buffer[1024] = {};
    // char Server::send_buffer[1024] = {};
    // int Server::listenfd = 0;
    // int Server::clintfd = 0;
    // int Server::iret = 0;

}  // namespace icg

#endif  // ICG_INCLUDE_ICG_SERVER_H_
