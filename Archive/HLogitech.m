%% Homography Shift
%Takes data output from SRVReader along with an image sequence from visuals
%camera and generates an overlay of virtual and/or real robot locations as
%a parallel image sequence.  

%% Variables

% Homographies
iHi = [0.040706   3.206330  -2321.272610
       2.179371  -0.027209  -1666.669287
       0.000004  -0.000483   1.000000000];
   
Hi  = [0.119175   0.298215   773.663980
       0.312750  -0.007101   714.143555
       0.000151  -0.000004   1.00000000];

%other variables
fileName = 'bmps\Movie_f0';
ext = 'bmp';
frames = 10;
fstart = 1370;
fskip=5;                    %only look at 1 out of every x frames
fps = 30/fskip;             %modified framerate
dCursor = 1;                %data cursor
tOffset = data{1,1}(1,5);   %initial offset in dataset
Nr = numel(data);           %Number of robots
loc = zeros(3, Nr);         %location of bots
or = zeros(2,Nr);           %orientation in XY unit vector components
rect = zeros(4, 3, Nr);     %rectangle points
rX=125;                     %rectangle size
rY=75;
tCursor = 0;                %time cursor

%% Frame Loop
for j = 1:frames
fprintf(['frame ' num2str(j) ' out of ' num2str(frames) '\n'])
ItestR = imread(strcat(fileName, num2str((j*fskip)+fstart-1), '.', ext));
Itest = imrotate(ItestR,180);
image(Itest);
hold on;

%% Homography Shift
time=(j-1)/fps;                             %time in seconds
while (tCursor < time)
    if(tCursor < time)
        dCursor=dCursor+1;
        tCursor=data{1,1}(dCursor, 5)-tOffset;
    end
end

%Interpolation Ratio
if (dCursor ~= 1)
    IR =(time-data{1,1}(dCursor-1, 5)+tOffset)/(tCursor-data{1,1}(dCursor-1, 5)+tOffset);
    for i = 1:Nr
        dat = [1000-data{1,i}(dCursor-1,1) 400-data{1,i}(dCursor-1,2)...
            data{1,i}(dCursor-1,6) data{1,i}(dCursor-1,7)]';         
        dat2 = [1000-data{1,i}(dCursor,1) 400-data{1,i}(dCursor,2)...
            data{1,i}(dCursor,6) data{1,i}(dCursor,7)]';
        dat(:)=dat(:)+IR.*(dat2(:)-dat(:));
        loc(1:2,i)=[dat(1);dat(2)];
        or(1:2,i)=[dat(3);dat(4)];
    end
else
     for i = 1:Nr
        dat = [1000-data{1,i}(dCursor,1) 400-data{1,i}(dCursor,2)...
            data{1,i}(dCursor,6) data{1,i}(dCursor,7)]';
        loc(1:2,i)=[dat(1);dat(2)];
        or(1:2,i)=[dat(3);dat(4)];
     end
end
for i = 1:Nr
    %coordinates for drawing orientation rectangles
    rect(1, 1:3,i)=Hi*[loc(1,i)+rX*or(1,i)+rY*or(2,i);loc(2,i)+rX*or(2,i)-rY*or(1,i);1]; %homography shift
    rect(2, 1:3,i)=Hi*[loc(1,i)-rX*or(1,i)+rY*or(2,i);loc(2,i)-rX*or(2,i)-rY*or(1,i);1]; %homography shift
    rect(3, 1:3,i)=Hi*[loc(1,i)-rX*or(1,i)-rY*or(2,i);loc(2,i)-rX*or(2,i)+rY*or(1,i);1]; %homography shift
    rect(4, 1:3,i)=Hi*[loc(1,i)+rX*or(1,i)-rY*or(2,i);loc(2,i)+rX*or(2,i)+rY*or(1,i);1]; %homography shift
    for k=1:4
        rect(k, 1:2,i)=rect(k, 1:2,i)./(rect(k,3,i).*2.5);    %scale picture
        rect(k, 2,i)=480-rect(k, 2,i);                        %mirror vertically
        rect(k, 1,i)=640-rect(k, 1,i);                        %mirror horizontally
    end
    temp=zeros(2,5);
    for k=1:4
        temp(1:2,k)=rect(k,1:2,i);
    end
    temp(1:2,5)=rect(1,1:2,i);
    plot(temp(1,:), temp(2,:), '-y ');
    
    %center of bot
    loc(1:3,i)=Hi*[loc(1,i);loc(2,i);1];        %homography shift
    loc(1:2,i)=loc(1:2,i)/(loc(3,i)*2.5);       %scale picture
    loc(2,i)=480-loc(2,i);                      %mirror vertically
    loc(1,i)=640-loc(1,i);                      %mirror horizontally
end
%plot center of robot
%plot(loc(1,:),loc(2,:),' co');


F2(j) = getframe;

%Save bitmaps
%P = frame2im(F2(j));
%directory = 'output/';
%number = num2str(j); 
%extension = '.bmp';
%filename = [directory number extension];
%imwrite(P,eval('filename'), 'bmp');
hold off;
end      
%movie2avi(F,'run.avi','fps',30,'compression', 'None');
fprintf(['getting video ready...\n'])
movie(F2, 1, 60);       %2X speed
