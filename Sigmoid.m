clear all
clc
load('in_MFCC.mat')
load('out10_MFCCe.mat')
target2file=target2file(:,2:size(target2file,2));
x=randperm(size(invector,2));
invector=invector(:,x);
target2file=target2file(:,x);
for i=1:size(invector,1)
    sum=0;
    for j=1:size(invector,2)
        sum=sum+invector(i,j);
    end
    mean(i)=sum/size(invector,2);
end
count=0;
for i=1:size(invector,1)
    sum=0;
    for j=1:size(invector,2)
        sum=sum+(invector(i,j)-mean(i))^2;
    end
    var(i)=sum/size(invector,2);
    if(var(i)>70.8)
        count=count+1;
        wineInputs2(count,:)=invector(i,:);
    end
end
wineTargets2=target2file;
min=9999999; max=-9999999;
for i=1:size(wineInputs2,1)
    for j=1:size(wineInputs2,2)
        if(min>wineInputs2(i,j))
            min=wineInputs2(i,j);
        end
        if(max<wineInputs2(i,j))
            max=wineInputs2(i,j);
        end
    end
    range=max-min;
    wineInputs2(i,:)=wineInputs2(i,:)/range;
end
n=5;
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
            weight(j,k,i-1)=randn;
        end
    end
    for j=1:layer(i)
        bias(j,i-1)=randn;
    end
end
te=10^-5;
preverror=10000;
iter=0;
eta=0.1;
count=0;
flag=0;flagm=0;
while(preverror>0.0001)
    error=0;
    weightdelt=zeros(size(weight));
    biasdelt=zeros(size(bias));
    for nte=1:2279
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
                otpt(j,i)=1/(1+exp(-inpt(j,i)));
            end
        end
        err=zeros(1,layer(n));
        for i=1:layer(n)
            error=error + (otptn(i)-otpt(i,n))^2;
            err(i)=otptn(i)-otpt(i,n);
        end  
        %2_layer_NN
        for j=1:layer(n-1)
            sum(j,n-1)=0;
            for k=1:layer(n)
                delta2=err(k)*otpt(k,n)*(1-otpt(k,n));
                weightdelt(j,k,n-1)=delta2*otpt(j,n-1);
                if(j==1)
                    biasdelt(k,n-1)=delta2;
                end
                sum(j,n-1)=sum(j,n-1)+delta2*weight(j,k,n-1);
            end
        end
        %3_layer_NN
        for i=1:layer(n-2)
            sum(i,n-2)=0;
            for j=1:layer(n-1)
                delta3=sum(j,n-1)*otpt(j,n-1)*(1-otpt(j,n-1));
                weightdelt(i,j,n-2)=delta3*otpt(i,n-2);
                if(i==1)
                    biasdelt(j,n-2)=delta3;
                end
                sum(i,n-2)=sum(i,n-2)+delta3*weight(i,j,n-2);
            end
        end
        
        %4_layer_NN
        for h=1:layer(n-3)
            sum(h,n-3)=0;
            for i=1:layer(n-2)
                delta4=sum(i,n-2)*otpt(i,n-2)*(1-otpt(i,n-2));
                weightdelt(h,i,n-3)=delta4*otpt(h,n-3);
                if(h==1)
                    biasdelt(i,n-3)=delta4;
                end
                sum(h,n-3)=sum(h,n-3)+delta4*weight(h,i,n-3);
            end
        end
        %5_layer_NN
        for g=1:layer(n-4)
            sum(g,n-4)=0;
            for h=1:layer(n-3)
                delta5=sum(h,n-3)*otpt(h,n-3)*(1-otpt(h,n-3));
                weightdelt(g,h,n-4)=delta5*otpt(g,n-4);
                if(g==1)
                    biasdelt(h,n-4)=delta5;
                end
                sum(g,n-4)=sum(g,n-4)+delta5*weight(g,h,n-4);
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
    fprintf('%.15f %.15f\n',error/(2279*layer(n)),preverror);
    if(preverror>error/(2279*layer(n)))
        count=0;flag=0;
        prevweight=weight;
        prevbias=bias;
        prevweightdelt=weightdelt;
        prevbiasdelt=biasdelt;
        preverror=error/(2279*layer(n));
        if(error<=te)
            disp('\nError Threshold Reached!!!!\n');
            break;
        end
        etam=eta;
        if(flagm==0)
            eta=eta+0.02;
        end
        iter=iter+1;
        disp(iter);
        for m=2279+1:size(wineInputs2,2)
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
                    otpt(j,i)=1/(1+exp(-inpt(j,i)));
                end
            end
            testotpt(:,m-2279)=otpt(:,n);
        end
        ans=zeros(layer(n),size(wineInputs2,2)-2279);
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
            if(wineTargets2(posi,2279+i)==1)
                storepers=storepers+1;
            end
        end
        storepers=storepers/size(testotpt,2);
        if(storeprevpers<storepers)
            storeweight=weight;
            storebias=bias;
        end
        plotconfusion(wineTargets2(:,2279+1:size(wineInputs2,2)),ans);
    else
        disp('\nPreverror<=Error!!!!\n');
        weight=prevweight;
        bias=prevbias;
        weightdelt=prevweightdelt;
        biasdelt=prevbiasdelt;
        if (flag==0)
            eta=eta/1.5;
            count=count+1;
            if (count==5)
                flag=1;
                flagm=1;
            end
        else
            eta=etam*1.15;
        end
    end
end