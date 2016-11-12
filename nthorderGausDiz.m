clear all
clc
load('in_MFCC.mat')
n=size(invector);
g=input('Enter Number of groups to be formed');
rangemean=1000*ones(n(1),2);
rangemean(:,2)=-1*rangemean(:,2);
minmidmax=zeros(n(1),g);
%{
for i=1:n(1)
    for j=1:n(2)
        if(invector(i,j)<rangemean(i,1))
            rangemean(i,1)=invector(i,j);
        end
        if(invector(i,j)>rangemean(i,3))
            rangemean(i,3)=invector(i,j);
        end
    end
    rangemean(i,2)=(rangemean(i,3)+rangemean(i,1))/2;
end
minmidmax=rangemean;
for iter=1:20
    for i=1:n(1)
        sum=zeros(g);
        arraycount=zeros(g,1);
        arraysort=zeros(g,n(2));
        dist=zeros(g,1);
        for j=1:n(2)
            for k=1:g
                dist(k)=abs(minmidmax(i,k)-invector(i,j));
            end
            minimum=dist(1);
            posi=1;
            for k=2:g
                if(dist(k)<minimum)
                    minimum=dist(k);
                    posi=k;
                end
            end
            arraycount(posi)=arraycount(posi)+1;
            arraysort(posi,arraycount(posi))=i;
            sum(posi)=sum(posi)+invector(i,j);
        end
        for j=1:g
            minmidmax(i,j)=sum(j)/arraycount(j);
        end
    end
end
A=minmidmax;


for i=1:n(1)
    for j=1:n(2)
        if(invector(i,j)<rangemean(i,1))
            rangemean(i,1)=invector(i,j);
        end
        if(invector(i,j)>rangemean(i,3))
            rangemean(i,3)=invector(i,j);
        end
    end
    rangemean(i,2)=(rangemean(i,3)+rangemean(i,1))/2;
    if (mod(g,2)~=0)
        a=-(g-1)/2:1:(g-1)/2;
    else
        a=-g+1:1:g-1;
    end
        minmidmax(i,:)=rangemean(i,2)+a(:)*10;
    for j=1:g
        minmidmax(i,j)=rangemean(i,1)+(rangemean(i,3)-rangemean(i,1))*(j-1)/(g-1);
    end
end
%}
for i=1:n(1)
    mean=0;
    for j=1:n(2)
        if(invector(i,j)<rangemean(i,1))
            rangemean(i,1)=invector(i,j);
        end
        if(invector(i,j)>rangemean(i,2))
            rangemean(i,2)=invector(i,j);
        end
        %mean=mean+invector(i,j);
    end
    %{
    mean=mean/n(2);
    minimum=min(abs(rangemean(i,1)-mean),abs(rangemean(i,2)-mean));
    minmidmax(i,1)=mean-minimum/2;
    minmidmax(i,2)=mean+minimum/2;
    %}
    minmidmax(i,2)=minmidmax(i,2)/n(2);
    minmidmax(i,1)=rangemean(i,1);
    minmidmax(i,3)=rangemean(i,2);
end
%{
for i=1:n(1)
    for j=1:59
        minmidmax(i,1)=minmidmax(i,1)+invector(i,j);
    end
    minmidmax(i,1)=minmidmax(i,1)/59;
    for j=60:130
        minmidmax(i,2)=minmidmax(i,2)+invector(i,j);
    end
    minmidmax(i,2)=minmidmax(i,2)/71;
    for j=130:178
        minmidmax(i,3)=minmidmax(i,3)+invector(i,j);
    end
    minmidmax(i,3)=minmidmax(i,3)/48;
end
%}
for iter=1:11
    sum=zeros(n(1),g);
    arraysort=zeros(g,n(2));
    arraycount=zeros(g,1);
    vartemp=zeros(n(1),g);
    var=zeros(n(1),g);
    std=zeros(n(1),g);
    for i=1:n(2)
        dist=zeros(g,1);
        for j=1:g
            for k=1:n(1)
                vartemp(k,j)=(minmidmax(k,j)-invector(k,i))^2;
                dist(j)=dist(j)+vartemp(k,j);
            end
            dist(j)=sqrt(dist(j));
        end
        minimum=dist(1);
        posi=1;
        for j=2:g
            if(dist(j)<minimum)
                minimum=dist(j);
                posi=j;
            end
        end
        arraycount(posi)=arraycount(posi)+1;
        arraysort(posi,i)=1;
        for j=1:n(1)
            sum(j,posi)=sum(j,posi)+invector(j,i);
            var(j,posi)=var(j,posi)+vartemp(j,posi);
        end
    end
    for i=1:g
        var(:,i)=var(:,i)/arraycount(i);
        std(:,i)=sqrt(var(:,i));
        minmidmax(:,i)=sum(:,i)/arraycount(i);
    end
    %{
    if (iter==19)
    minmidmax9=minmidmax;
    end
    if (iter==18)
    minmidmax8=minmidmax;
    end
    if (iter==17)
    minmidmax7=minmidmax;
    end
    if (iter==16)
    minmidmax6=minmidmax;
    end
    if (iter==15)
    minmidmax5=minmidmax;
    end
    if (iter==14)
    minmidmax4=minmidmax;
    end
    %}
end

gauspdis=ones(g,n(2));
prob=ones(g,n(2));
for i=1:n(2)
    sum1=0;
    for j=1:g
        for k=1:n(1)
            gauspdis(j,i)=gauspdis(j,i)*exp((-(invector(k,i)-minmidmax(k,j))^2)/(2*var(k,j)))/(sqrt(2*pi)*std(k,j));
        end
        sum1=sum1+gauspdis(j,i);
    end
    prob(:,i)=gauspdis(:,i)/sum1;
end
%fprintf('%.16f',prob(1,1));
