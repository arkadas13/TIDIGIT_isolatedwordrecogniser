clear all
clc
wrong=0;
right=0;
teach=0;
val=exp(-1/2);
while(teach<10)
    n=5;%n=input('Enter No. of layers:');y=cos(1.232*(x-1)).*exp(-((x-1)*0.704).^2/2) +ve,,, cos(1.232*(x-1)).*exp(-((x-1)*0.704).^2/2)/0.289 -ve;diff=-sin(1.232*(x-1))/1.232.*exp(-((x-1)*0.704)
    layer=zeros(size(n));
    max=0;
    scale=1000000;
    for i=1:n
        layer(i)=randi([10,20],1,1);
        if(i~=1)
            for j=1:layer(i-1)
                for k=1:layer(i)
                    if(mod(randi([1,1000],1,1),2)==1)
                        weight(j,k,i-1)=rand()/scale;
                    else
                        weight(j,k,i-1)=-1*rand()/scale;
                    end
                end
            end
            for j=1:layer(i)
                if(mod(randi([1,1000],1,1),2)==1)
                    bias(j,i-1)=rand()/scale;
                else
                    bias(j,i-1)=-1*rand()/scale;
                end
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
        inpt(i,1)=randi([-99,99]);%input('Enter training data:')/scale;
    end
    otpt=inpt(:,1);
    for i=1:layer(n)
        if(mod(randi([1,1000],1,1),2)==1)
            otptn(i)=rand();
        else
            otptn(i)=-1*rand();
        end%input('Enter target data:');
    end
    disp(inpt);
    disp(otptn);
    %{
inpt(:,1)=[.05;0.1];
otpt(:,1)=inpt(:,1);
otptn=[0.01;0.99];
    %}
    preverror=100;
    te=10^-10;                              %Input threshold error
    for iter=1:1000
        for i=2:n
            for j=1:layer(i)
                inpt(j,i)=0;
                for k=1:layer(i-1)
                    inpt(j,i)=inpt(j,i)+weight(k,j,i-1)*otpt(k,i-1);
                end
                inpt(j,i)=inpt(j,i)+bias(j,i-1);
                if(inpt(j,i)>1 || inpt(j,i)<-1)
                    fprintf('****>>>>>>>> %.20f\n',inpt(j,i));
                end
                otpt(j,i)=inpt(j,i)*exp(-(inpt(j,i)^2)/2)/val;
            end
        end
        %disp(inpt);
        %disp(otpt);
        err=zeros(1,layer(n));
        error=0;
        for i=1:layer(n)
            error=error + (otptn(i)-otpt(i,n))^2;
            err(i)=err(i)+otptn(i)-otpt(i,n);
        end
        if(preverror>=error/layer(n))
            fprintf('%.20f %.20f\n',error/layer(n),preverror);
            preverror=error/layer(n);
            if(error<=te)
                disp('Error Threshold Reached!!!!');
                right=right+1;
                break;
            end
        else
            disp('Preverror<=Error!!!!');
            wrong=wrong+1;
            break;
        end
        
        weightdelt=zeros(size(weight));
        biasdelt=zeros(size(bias));
        eta=0.5;
        %2_layer_NN
        for j=1:layer(n-1)
            sum(j,n-1)=0;
            for k=1:layer(n)
                delta2=err(k)*exp(-(inpt(k,n)^2)/2)*(1-inpt(k,n)^2)/val;
                weightdelt(j,k,n-1)=delta2*otpt(j,n-1);
                biasdelt(k,n-1)=delta2;
                sum(j,n-1)=sum(j,n-1)+delta2*weight(j,k,n-1)/val;
            end
        end
        %3_layer_NN
        for i=1:layer(n-2)
            sum(i,n-2)=0;
            for j=1:layer(n-1)
                delta3=sum(j,n-1)*exp(-(inpt(j,n-1)^2)/2)*(1-inpt(j,n-1)^2)/val;
                weightdelt(i,j,n-2)=delta3*otpt(i,n-2);
                biasdelt(j,n-2)=delta3;
                sum(i,n-2)=sum(i,n-2)+delta3*weight(i,j,n-2)/val;
            end
        end
        %}
        %4_layer_NN
        for h=1:layer(n-3)
            sum(h,n-3)=0;
            for i=1:layer(n-2)
                delta4=sum(i,n-2)*exp(-(inpt(h,n-2)^2)/2)*(1-inpt(h,n-2)^2)/val;
                weightdelt(h,i,n-3)=delta4*otpt(h,n-3);
                biasdelt(i,n-3)=delta4;
                sum(h,n-3)=sum(h,n-3)+delta4*weight(h,i,n-3)/val;
            end
        end
        %5_layer_NN
        for g=1:layer(n-4)
            sum(g,n-4)=0;
            for h=1:layer(n-3)
                delta5=sum(h,n-3)*exp(-(inpt(g,n-3)^2)/2)*(1-inpt(g,n-3)^2)/val;
                weightdelt(g,h,n-4)=delta5*otpt(g,n-4);
                biasdelt(h,n-4)=delta5;
                sum(g,n-4)=sum(g,n-4)+delta4*weight(g,i,n-4)/val;
            end
        end
        %weight_bias_updation
        for k=1:n-1
            for i=1:layer(k+1)
                for j=1:layer(k)
                    weight(j,i,k)=weight(j,i,k)+eta*weightdelt(j,i,k);
                end
                bias(i,k)=bias(i,k)+eta*biasdelt(i,k);
            end
        end
        %fprintf('>>>>>>>>>>%.20f %.20f\n',inpt(2,1),otpt(2,1))
        if(iter==1000)
            right=right+1;
        end
    end
    teach=teach+1;
end