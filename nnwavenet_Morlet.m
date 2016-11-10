clear all
clc
wrong=0;
right=0;
teach=0;
while(teach<10)
    n=3;%n=input('Enter No. of layers:');y=cos(1.232*(x-1)).*exp(-((x-1)*0.704).^2/2) +ve,,, cos(1.232*(x-1)).*exp(-((x-1)*0.704).^2/2)/0.289 -ve;diff=-sin(1.232*(x-1))/1.232.*exp(-((x-1)*0.704)
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
                otpt(j,i)=cos(1.232*(inpt(j,i)-1))*exp(-((inpt(j,i)-1)*0.5)^2);
                if(otpt(j,i)<0)
                    otpt(j,i)=otpt(j,i)/0.289;
                end
                %{
            min=100;
            %disp(inpt(j,i));
            if(inpt(j,i)>1 || inpt(j,i)<-1)
                %disp('************')
                for x=-1:0.000005:1
                    flag=0;
                    diff=abs(val2-x*exp(-(x^2)/2));
                    if(diff<min)
                        min=diff;
                        x1=x;
                        flag=1;
                    end
                    if(flag==0)
                        break;
                    end
                end
                frac=x1/inpt(j,i);
            else
                x1=inpt(j,i);
                frac=1;
            end
            otpt(j,i)=x1*exp(-(x1^2)/2);
            weight(:,j,i-1)=weight(:,j,i-1)*frac;
            fprintf('>>>%.20f %.20f %.20f %.20f\n',x1,inpt(j,i),val2,otpt(j,i));
            inpt(j,i)=x1;
            bias(j,i-1)=bias(j,i-1)*frac;
                %}
            end
        end
        %disp(inpt);
        %disp(otpt);
        err=zeros(1,layer(n));
        error=0;
        for i=1:layer(n)
            error=error + (otptn(i)-otpt(i,n))^2;
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
        for i=1:layer(n)
            err(i)=err(i)+otptn(i)-otpt(i,n);
        end
        weightdelt=zeros(size(weight));
        biasdelt=zeros(size(bias));
        eta=0.5;
        %2_layer_NN
        for j=1:layer(n-1)
            for k=1:layer(n)
                delta2=err(k)*-exp(-((inpt(j,n)-1)^2)*0.25)*(1.232*sin(1.232*(inpt(j,n)-1))+0.25*(inpt(j,n)-1)*cos(1.232*(inpt(j,n)-1)));
                if(otpt(j,n)<0)
                    delta2=delta2/0.289;
                end
                weightdelt(j,k,n-1)=delta2*otpt(j,n-1);
                biasdelt(k,n-1)=delta2;
            end
        end
        %3_layer_NN
        for i=1:layer(n-2)
            for j=1:layer(n-1)
                sum0=0;
                for k=1:layer(n)
                    x=err(k)*-exp(-((inpt(j,n)-1)^2)*0.25)*(1.232*sin(1.232*(inpt(j,n)-1))+0.25*(inpt(j,n)-1)*cos(1.232*(inpt(j,n)-1)))*weight(j,k,n-1);
                    if(otpt(j,n)<0)
                        x=x/0.289;
                    end
                    sum0=sum0+x;
                end
                delta3=sum0*-err(k)*exp(-((inpt(j,n-1)-1)^2)*0.25)*(1.232*sin(1.232*(inpt(j,n-1)-1))+0.25*(inpt(j,n-1)-1)*cos(1.232*(inpt(j,n-1)-1)));
                if(otpt(j,n-1)<0)
                    delta3=delta3/0.289;
                end
                weightdelt(i,j,n-2)=delta3*otpt(i,n-2);
                biasdelt(j,n-2)=delta3;
            end
        end
        %{
    4_layer_NN
    for h=1:layer(n-3)
        sum=0;
        for i=1:layer(n-2)
            sum1=0;
            for j=1:layer(n-1)
                sum0=0;
                for k=1:layer(n)
                    sum0=sum0+exp(-(inpt(j,n)^2)/2)*(1-inpt(j,n)^2)*weight(j,k,n-1);
                end
                sum1=sum1+sum0*exp(-(inpt(j,n-1)^2)/2)*(1-inpt(j,n-1)^2)*weight(j,k,n-2);
            end
            delta4=sum1*exp(-(inpt(j,n-2)^2)/2)*(1-inpt(j,n-2)^2);
            weightdelt(h,i,n-3)=delta4*otpt(h,n-3);
            biasdelt(i,n-3)=delta4;
        end
    end
    %5_layer_NN
    for g=1:layer(n-4)
        for h=1:layer(n-3)
            sum2=0;
            for i=1:layer(n-2)
                sum1=0;
                for j=1:layer(n-1)
                    sum0=0;
                    for k=1:layer(n)
                        sum0=sum0+err(k)*exp(-(inpt(j,n)^2)/2)*(1-inpt(j,n)^2)*weight(j,k,n-1);
                    end
                    sum1=sum1+sum0*exp(-(inpt(j,n-1)^2)/2)*(1-inpt(j,n-1)^2)*weight(j,k,n-2);
                end
                sum2=sum2+sum1*exp(-(inpt(j,n-2)^2)/2)*(1-inpt(j,n-2)^2)*weight(j,k,n-3);
            end
            delta5=sum2*exp(-(inpt(j,n-3)^2)/2)*(1-inpt(j,n-3)^2);
            weightdelt(g,h,n-4)=delta5*otpt(g,n-4);
            biasdelt(h,n-4)=delta5;
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
        %fprintf('>>>>>>>>>>%.20f %.20f\n',inpt(2,1),otpt(2,1))
        if(iter==1000)
            right=right+1;
        end
    end
    teach=teach+1;
end