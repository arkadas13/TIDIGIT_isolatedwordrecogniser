clear all
clc
load('in_MFCC.mat')
load('out10_MFCCe.mat')
target2file=target2file(:,2:size(target2file,2));
x=randperm(size(invector,2));
invector=invector(:,x);
target2file=target2file(:,x);
scale=1;val=exp(-0.5); 
sf=500;
min=9999999; max=-9999999;
for i=1:size(invector,1)
    for j=1:size(invector,2)
        if(min>invector(i,j))
            min=invector(i,j);
        end
        if(max<invector(i,j))
            max=invector(i,j);
        end
    end
    range=max-min;
    invector(i,:)=invector(i,:)/(range);
end
for i=1:size(invector,1)
    sum1=0;
    for j=1:size(invector,2)
        sum1=sum1+invector(i,j);
    end
    mean(i)=sum1/size(invector,2);
end
count=0;
for i=1:size(invector,1)
    sum1=0;
    for j=1:size(invector,2)
        sum1=sum1+(invector(i,j)-mean(i))^2;
    end
    var(i)=sum1/size(invector,2);
    if(var(i)>0.00115)
        count=count+1;
        wineInputs2(count,:)=invector(i,:);
    end
end
for i=1:size(target2file,2)
    for j=1:size(target2file,1)
        if(target2file(j,i)==0)
            target2file(j,i)=-1;
        end
    end
end
wineTargets2=target2file;
n=3;
layer=zeros(n,1);
max=0;
layer(1)=size(wineInputs2,1);
for i=2:n
    if(i~=n)
        layer(i)=input('Enter number of nodes in layer');
    else
        layer(i)=size(wineTargets2,1);
    end
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
te=10^-5;
preverror=10000;
iter=0;
eta=150;
count=0;
flag=0;flagm=0;
while(preverror>0.0001)
    error=0;
    weightdelt=zeros(size(weight));
    biasdelt=zeros(size(bias));
    %f=0;
    for nte=1:3650
        inpt(:,1)=wineInputs2(:,nte);
        otpt(:,1)=inpt(:,1);
        otptn=wineTargets2(:,nte);
        for i=2:n
            for j=1:layer(i)
                inpt(j,i)=0;
                for k=1:layer(i-1)
                    inpt(j,i)=inpt(j,i)+weight(k,j,i-1)*otpt(k,i-1);
                end
                inpt(j,i)=inpt(j,i)+bias(j,i-1);
                if(inpt(j,i)>1 || inpt(j,i)<-1)
                    %fprintf('>>>>>>>> %.20f\n',inpt(j,i));
                    %f=1;
                    %break;
                end
                otpt(j,i)=(inpt(j,i)/sf)*exp(-((inpt(j,i)/sf)^2)/2)/val;
            end
%             if(f==1)
%                 break;
%             end
        end
%         if(f==1)
%             break;
%         end
        err=zeros(1,layer(n));
        for i=1:layer(n)
            error=error + (otptn(i)-otpt(i,n))^2;
            err(i)=otptn(i)-otpt(i,n);
        end  
        %2_layer_NN
        for j=1:layer(n-1)
            sum(j,n-1)=0;
            for k=1:layer(n)
                delta2=err(k)*exp(-((inpt(k,n)/sf)^2)/2)*(1-(inpt(k,n)/sf)^2)/(sf*val);
                weightdelt(j,k,n-1)=delta2*otpt(j,n-1);
                biasdelt(k,n-1)=delta2;
                sum(j,n-1)=sum(j,n-1)+delta2*weight(j,k,n-1);
            end
        end
        %3_layer_NN
        for i=1:layer(n-2)
            sum(i,n-2)=0;
            for j=1:layer(n-1)
                delta3=sum(j,n-1)*exp(-((inpt(j,n-1)/sf)^2)/2)*(1-(inpt(j,n-1)/sf)^2)/(sf*val);
                weightdelt(i,j,n-2)=delta3*otpt(i,n-2);
                biasdelt(j,n-2)=delta3;
                sum(i,n-2)=sum(i,n-2)+delta3*weight(i,j,n-2);
            end
        end
        %{
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
        %}
        for k=1:n-1
            for i=1:layer(k+1)
                for j=1:layer(k)
                    weight(j,i,k)=weight(j,i,k)+eta*weightdelt(j,i,k);
                end
                bias(i,k)=bias(i,k)+eta*biasdelt(i,k);
            end
        end
    end
    q=error/(3650*layer(n));
    fprintf('%.15f %.15f\n',q,preverror);
    if(preverror>q)
        count=0;flag=0;
        prevweight=weight;
        prevbias=bias;
        prevweightdelt=weightdelt;
        prevbiasdelt=biasdelt;
        preverror=q;
        if(error<=te)
            disp('\nError Threshold Reached!!!!\n');
            break;
        end
        etam=eta;
%         if(flagm==0)
             eta=eta*2.1;
%         end
        iter=iter+1;
        disp(iter);
        for m=3650+1:size(wineInputs2,2)
            for i=1:layer(1)
                inpt(i,1)=wineInputs2(i,m);
            end
            otpt(:,1)=inpt(:,1);
            for i=2:n
                for j=1:layer(i)
                    inpt(j,i)=0;
                    for k=1:layer(i-1)
                        inpt(j,i)=inpt(j,i)+weight(k,j,i-1)*otpt(k,i-1);
                    end
                    inpt(j,i)=inpt(j,i)+bias(j,i-1);
%                     if(inpt(j,i)>1 || inpt(j,i)<-1)
%                         fprintf('****>>>>>>>> %.20f\n',inpt(j,i));
%                     end
                    otpt(j,i)=(inpt(j,i)/sf)*exp(-((inpt(j,i)/sf)^2)/2)/val;
                end
            end
            testotpt(:,m-3650)=otpt(1:layer(n),n);
        end
        ans=-1*ones(layer(n),size(wineInputs2,2)-3650);
        storepers=0;storeprevpers=0;
        for i=1:size(testotpt,2)
            max=testotpt(1,i);
            posi=1;
            for j=2:size(testotpt,1)
                if(max<testotpt(j,i))
                    max=testotpt(j,i);
                    posi=j;
                end
            end
            ans(posi,i)=1;
            if(wineTargets2(posi,3650+i)==1)
                storepers=storepers+1;
            end
        end
        storepers=storepers/size(testotpt,2);
        disp(storepers);
        if(storeprevpers<storepers)
            storeprevpers=storepers;
            storeweight=weight;
            storebias=bias;
            storeiter=iter;
        end
        plotconfusion(wineTargets2(:,3650+1:size(wineInputs2,2)),ans);
    else
        disp('\nPreverror<=Error!!!!\n');
        weight=prevweight;
        bias=prevbias;
        weightdelt=prevweightdelt;
        biasdelt=prevbiasdelt;
%          if (flag==0)
             eta=eta/10;
%             count=count+1;
%              if (count==5)
%                  flag=1;
%                  flagm=1;
%              end
%         else
%             eta=etam*1.15;
%         end
    end
end