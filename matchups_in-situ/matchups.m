function sate = matchups(date, obs_c, window_t)
variables = ["Rrs_412","Rrs_443","Rrs_469",...
    "Rrs_488","Rrs_531","Rrs_547","Rrs_555",...
    "Rrs_645","Rrs_667","Rrs_678",...
    "par","sst","sstn","sst4"];

sate = nan(length(variables),window_t);
for t = 1:window_t
    for i = 1:length(variables)
        file = nc_file(date-t+1, variables(i));
        if file ~= ""
            sate_3x3 = obs_match_file(file,variables(i),...
                obs_c,3);
            sate(i,t) = nanmean(sate_3x3(:));
        end
    end
    
end
end

%%
function file = nc_file(date, vari)
switch vari
    case "Rrs_412"
        subpath = "Remote sensing reflectance at 412 nm";
        str1 = ".L3m.DAY.RRS.Rrs_412.4km.nc";
    case "Rrs_443"
        subpath = "Remote sensing reflectance at 443 nm";
        str1 = ".L3m.DAY.RRS.Rrs_443.4km.nc";
    case "Rrs_469"
        subpath = "Remote sensing reflectance at 469 nm";
        str1 = ".L3m.DAY.RRS.Rrs_469.4km.nc";
    case "Rrs_488"
        subpath = "Remote sensing reflectance at 488 nm";
        str1 = ".L3m.DAY.RRS.Rrs_488.4km.nc";
    case "Rrs_531"
        subpath = "Remote sensing reflectance at 531 nm";
        str1 = ".L3m.DAY.RRS.Rrs_531.4km.nc";
    case "Rrs_547"
        subpath = "Remote sensing reflectance at 547 nm";
        str1 = ".L3m.DAY.RRS.Rrs_547.4km.nc";
    case "Rrs_555"
        subpath = "Remote sensing reflectance at 555 nm";
        str1 = ".L3m.DAY.RRS.Rrs_555.4km.nc";
    case "Rrs_645"
        subpath = "Remote sensing reflectance at 645 nm";
        str1 = ".L3m.DAY.RRS.Rrs_645.4km.nc";
    case "Rrs_667"
        subpath = "Remote sensing reflectance at 667 nm";
        str1 = ".L3m.DAY.RRS.Rrs_667.4km.nc";
    case "Rrs_678"
        subpath = "Remote sensing reflectance at 678 nm";
        str1 = ".L3m.DAY.RRS.Rrs_678.4km.nc";
    case "par"
        subpath = "Photosynthetically Available Radiation";
        str1 = ".L3m.DAY.PAR.par.4km.nc";
    case "sst"
        subpath = "Sea Surface Temperature (11 u daytime)";
        str1 = ".L3m.DAY.SST.sst.4km.nc";
    case "sstn"
        subpath = "Sea Surface Temperature (11 u nighttime)";
        str1 = ".L3m.DAY.NSST.sst.4km.nc";
    case "sst4"
        subpath = "Sea Surface Temperature (4 u nighttime)";
        str1 = ".L3m.DAY.SST4.sst4.4km.nc";
        
end


file1 = sprintf("I:/MODIS-Aqua/"+subpath+"/AQUA_MODIS.%04d%02d%02d"+str1,...
    year(date),month(date),day(date));


if exist(file1, 'file')
    file = file1;
else
    file = "";
end
end










%%
function value = obs_match_file(pathfile,vari,obs,w)
if vari=="sstn"
   vari="sst"; 
end
if abs(obs(1))>90||abs(obs(2))>180
    error("经纬度超过界限！");
end
dx = 1/24;
% lat = 90-dx/2:dx:-90+dx/2;
% lon = -180+dx/2:dx:180-dx/2;
% [~,idx]=min(abs(lat-a))
% [~,idx]=min(abs(lon-b))
lon_max_index = 8640;
lat_max_index = 4320;

lat_idx = round(lat_max_index-(obs(1)+(90+dx/2))*1/dx+1);
lon_idx = round((obs(2)+(180+dx/2))*1/dx);
if lon_idx>lon_max_index
    lon_idx = lon_max_index;
end


dw = round((w-1)/2);
if dw==0
    value = ncread(pathfile,vari,[lon_idx-dw,lat_idx-dw],[w,w,]);
elseif (lat_idx-dw<=0)||(lat_idx-dw>lat_max_index)
    warning("高纬度匹配nan！");
    value = nan(w,w);
elseif lon_idx-dw<=0
    
    tmp1 = ncread(pathfile,vari,[lon_max_index+lon_idx-dw,lat_idx-dw],[inf,w]);
    tmp2 = ncread(pathfile,vari,[1,lat_idx-dw],[lon_idx+dw,w]);
    value = vertcat(tmp1,tmp2);
elseif lon_idx+dx>lon_max_index
    tmp1 = ncread(pathfile,vari,[lon_idx-dw,lat_idx-dw],[inf,w]);
    tmp2 = ncread(pathfile,vari,[1,lat_idx-dw],[lon_max_index-lon_idx+dw,w]);
    
    value = vertcat(tmp1,tmp2);
else
    value = ncread(pathfile,vari,[lon_idx-dw,lat_idx-dw],[w,w]);
end

end