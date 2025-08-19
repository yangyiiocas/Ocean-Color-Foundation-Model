close all;clear,clc
window_t = 30;


path="../²ÉÑùdataset/";
files = dir(path+"/*[pp]_(oppwgnew).txt");

lats= [];   lons = [];  dates = [];
m = length(files);
for i = 1:m
    
    txt = readtable(files(i).folder+"/"+files(i).name);
    date = txt{:,1};
    lat = txt{:,2};
    lon = txt{:,3};
    
    dates = [dates;date];
    lats = [lats;lat];
    lons = [lons;lon];
end

anchors = zeros(length(dates),5);
anchors(:,1) = year(dates);
anchors(:,2) = month(dates);
anchors(:,3) = day(dates);
anchors(:,4) = round(lats*24)/24;
anchors(:,5) = round(lons*24)/24;

anchors= unique(anchors, 'rows');

date_tmp = datetime(anchors(:,1),anchors(:,2),anchors(:,3));

m = length(date_tmp);


anchors_all = zeros(m*window_t,5);
for i=0:window_t-1
    anchors_all(i*m+1:(i+1)*m,1)=year(date_tmp-i);
    anchors_all(i*m+1:(i+1)*m,2)=month(date_tmp-i);
    anchors_all(i*m+1:(i+1)*m,3)=day(date_tmp-i);
    anchors_all(i*m+1:(i+1)*m,4)=anchors(:,4);
    anchors_all(i*m+1:(i+1)*m,5)=anchors(:,5);
end

anchors= unique(anchors_all, 'rows');
clear anchors_all date_tmp lat lats lon lons date dates txt;
%%
dates = datetime(anchors(:,1),anchors(:,2),anchors(:,3));
lat = anchors(:,4);    lon = anchors(:,5);

m = length(dates);
sate = zeros(m,14);
tic
parfor k = 1:m
    if dates(k)>=datetime(2002,7,4)
        sate(k,:) = matchups(dates(k),[lat(k),lon(k)],1);
    else
        sate(k,:) = nan(1,14);
    end
    fprintf("[%06d/%06d]\n",k,m);
end
toc
save("../Ò£¸Ðsat_matchups/sate.mat",'sate','dates','lat','lon');

% run obs_matchup.m





