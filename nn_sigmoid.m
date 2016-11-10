clear all
clc
n=input('Enter No. of layers:');
layer=zeros(n,1);
max=0;
for i=1:n
    layer(i)=input('Enter number of nodes in layer');
    if(i~=1)
        for j=1:layer(i-1)
            for k=1:layer(i)
                weight(j,k,i-1)=randn;
            end
        end
        for j=1:layer(i)
            bias(j,i-1)=randn;
        end
    end
end
disp(weight)
disp(bias)
%{
weight(:, :, 1)=[.15 .25; .2 .3];
weight(:,:,2)=[.4 .5;.45 .55];
bias=[0.35, 0.6;0.35, 0.6];
%}
for i=1:layer(1)
    inpt(i,1)=input('Enter training data:');
end
otpt(:,1)=inpt(:,1);
for i=1:layer(n)
    otptn(i)=input('Enter target data:');
end
%{
inpt(:,1)=[.05;0.1];
otpt(:,1)=inpt(:,1);
otptn=[0.01;0.99];
%}
preverror=1;
te=10^-18;                              %Input threshold error
for iter=1:1000
    for i=2:n
        for j=1:layer(i)
            inpt(j,i)=0;
            for k=1:layer(i-1)
                inpt(j,i)=inpt(j,i)+weight(k,j,i-1)*otpt(k,i-1);
            end
            inpt(j,i)=inpt(j,i)+bias(j,i-1);
            otpt(j,i)=1/(1+exp(-inpt(j,i)));
        end
    end
    err=zeros(1,layer(n));
    error=0;
    for i=1:layer(n)
        error=error + (otptn(i)-otpt(i,n))^2;
    end
    if(preverror>=error/layer(n))
        preverror=error/layer(n);
        fprintf('%.20f\n',error/layer(n));
        if(error<=te)
            disp('\nError Threshold Reached!!!!\n');
            break;
        end
    else
        disp('\nPreverror<=Error!!!!\n');
        break;
    end
    for i=1:layer(n)
        err(i)=err(i)+otptn(i)-otpt(i,n);
    end
    weightdelt=zeros(size(weight));
    biasdelt=zeros(size(bias));
    eta=0.1;
    %2_layer_NN
    for j=1:layer(n-1)
        sum(j,n-1)=0;
        for k=1:layer(n)
            delta2=err(k)*otpt(k,n)*(1-otpt(k,n));
            weightdelt(j,k,n-1)=delta2*otpt(j,n-1);
            biasdelt(k,n-1)=delta2;
            sum(j,n-1)=sum(j,n-1)+delta2*weight(j,k,n-1);
        end
    end
   %3_layer_NN
    for i=1:layer(n-2)
        sum(i,n-2)=0;
        for j=1:layer(n-1)
            delta3=sum(j,n-1)*otpt(j,n-1)*(1-otpt(j,n-1));
            weightdelt(i,j,n-2)=delta3*otpt(i,n-2);
            biasdelt(j,n-2)=delta3;
            sum(i,n-2)=sum(i,n-2)+delta3*weight(i,j,n-2);
        end
    end
    %}
    
    %4_layer_NN
    for h=1:layer(n-3)
        sum(h,n-3)=0;
        for i=1:layer(n-2)
            delta4=sum(i,n-1)*otpt(i,n-2)*(1-otpt(i,n-2));
            weightdelt(h,i,n-3)=delta4*otpt(h,n-3);
            biasdelt(i,n-3)=delta4;
            sum(h,n-3)=sum(h,n-3)+delta4*weight(h,i,n-3);
        end
    end
    %5_layer_NN
    for g=1:layer(n-4)
        sum(g,n-4)=0;
        for h=1:layer(n-3)
            delta5=sum(h,n-3)*otpt(h,n-3)*(1-otpt(h,n-3));
            weightdelt(g,h,n-4)=delta5*otpt(g,n-4);
            biasdelt(h,n-4)=delta5;
            sum(g,n-4)=sum(g,n-4)+delta4*weight(g,i,n-4);
        end
    end
    %}
    %weight_bias_updation
    for k=1:n-1
        for i=1:layer(k+1)
            for j=1:layer(k)
                weight(j,i,k)=weight(j,i,k)+eta*weightdelt(j,i,k);
            end
            bias(i,k)=bias(i,k)+eta*biasdelt(i,k);
        end
    end
end