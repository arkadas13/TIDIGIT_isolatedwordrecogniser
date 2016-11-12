clc
clear all
[f,Fs]=audioread('recordings\1_jackson_0.wav');
count=1;
x=0;
framelength=256;    %frame:32ms=0.032x8000=256samples%==========>user input
overlaplength=100;  %Overlap:12.5ms=0.0125x8000=100samples%=====>user input
for i=2:length(f)
    f(i)=f(i)-0.95*f(i-1);      %pre-emphasis
end
while(x+framelength<=length(f)) %framing
    sample(:,count)=f(x+1:x+framelength);
    x=x+framelength-overlaplength;
    count=count+1;
end
sample(:,count)=[f(x+1:length(f));zeros(framelength-length(f)+x,1)];

%{
figure(1)
plot(sample(:,1),'r')
hold on
original=sample(:,1);
%}

for i=1:count
    for j=1:framelength
        samplewin(j,i)=sample(j,i)*(0.54-0.46*cos(2*pi*(j-1)/(framelength-1)));%hamming window
    end
end

%{
plot(samplewin(:,1),'b')
hold off
 
figure(3)
x=1:framelength;
y=0.54-0.46*cos(2*pi*(x-1)/(framelength-1));
plot(x,y);
 
figure(2)
freq1=1:Fs;
freq2=1:Fs;
t=(1:256)'/Fs;
x=fft(sample(:,1), Fs);
y=fft(samplewin(:,1), Fs);
subplot(3,2,1); plot(t, sample(:,1)); grid on; axis([-inf inf -0.025 0.025]); title('Original signal');
subplot(3,2,2); plot(t, samplewin(:,1)); grid on; axis([-inf inf -0.025 0.025]); title('Windowed signal');
subplot(3,2,3); plot(freq1(1:Fs/2), abs(x(1:Fs/2))); grid on; title('Energy spectrum (linear scale)');
subplot(3,2,4); plot(freq2(1:Fs/2), abs(y(1:Fs/2))); grid on; title('Energy spectrum (linear scale)');
subplot(3,2,5); plot(freq1(1:Fs/2), 20*log10(abs(x(1:Fs/2)))); grid on; axis([-inf inf -100 10]); title('Energy spectrum (db)');
subplot(3,2,6); plot(freq2(1:Fs/2), 20*log10(abs(y(1:Fs/2)))); grid on; axis([-inf inf -100 10]); title('Energy spectrum (db)');
%}

for i=1:count
    samplefft(:,i)=fft(samplewin(:,i),framelength);
    periodogram(:,i)=(abs(samplefft(:,i)).^2)./framelength;
end
plot(periodogram(1:framelength/2,1))

minfreq=0;%====================================================>user input
maxfreq=2595*log10(1+(framelength/2+1)/(700));
nooffilt=26;%==================================================>user input
melfreq=minfreq:(maxfreq-minfreq)/(nooffilt+1):maxfreq;
hertzfreq=round(700.*(10.^(melfreq/2595)-1));
trifiltbank=zeros(framelength/2+1,nooffilt);

for i=1:nooffilt
    trifiltbank(hertzfreq(i)+1:hertzfreq(i+2)-1,i)=triang(hertzfreq(i+2)-hertzfreq(i)-1);
end

%plot(trifiltbank)

periodtrifilt=zeros(framelength/2+1,nooffilt,count);
melfiltpower=zeros(count,nooffilt);
logmelfiltpower=zeros(count,nooffilt);
for i=1:count
    for j=1:nooffilt
        sum=0;
        for k=1:framelength/2+1
            periodtrifilt(k,j,i)=periodogram(k,i)*trifiltbank(k,j);
            sum=sum+periodtrifilt(k,j,i);
        end
        melfiltpower(i,j)=melfiltpower(i,j)+sum;
        logmelfiltpower(i,j)=log10(melfiltpower(i,j));
    end
end
dct=zeros(count,nooffilt);
for i=1:count
    dct(i,:)=dct2(logmelfiltpower(i,:));
end
plot(dct(:,1:12)','b')