close all; clear,clc
win_t = 30;

usate = load("../遥感sat_matchups/sate.mat",'sate','dates','lat','lon');
for i = 1:length(usate.lat)
    anchor = sprintf("a%04da%02da%02da%04da%04d",...
        year(usate.dates(i)),...
        month(usate.dates(i)),...
        day(usate.dates(i)),...
        round((usate.lat(i)+90)*24),...
        round((usate.lon(i)+180)*24));
    database.(anchor) = usate.sate(i,:);
    
end

% files = dir("../采样dataset/*.txt");
files = dir("../采样dataset/*[pp]_(oppwgnew).txt");
m = length(files);
for i = 1:m
    txt = readtable(files(i).folder+"/"+files(i).name);
    dates = txt{:,1};
    lats = txt{:,2};
    lons = txt{:,3};    
    
    n = length(dates);
    sate = zeros(n,14,win_t);
    for k1=1:n
        for k2=1:win_t
            anchor = sprintf("a%04da%02da%02da%04da%04d",...
                year(dates(k1)-k2+1),...
                month(dates(k1)-k2+1),...
                day(dates(k1)-k2+1),...
                round((round(lats(k1)*24)/24+90)*24),...
                round((round(lons(k1)*24)/24+180)*24));
            sate(k1,:,k2) = database.(anchor); % %#ok<PFBNS>
        end
        fprintf("[%06d/%06d], [%06d/%06d]\n",i,m,k1,n);
    end
    parsave("../遥感sat_matchups/"+replace(files(i).name,".txt",".mat"), sate);
end

function parsave(fname, sate)
save(fname, 'sate', '-v7.3');
end








