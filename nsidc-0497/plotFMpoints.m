
% Script: plotFMpoints
%
% This script supports plotting of features included in the 'MOA-derived
% Structural Feature Map of the Ronne-Filchner Ice Shelf' data set.
%
% The features are stored with set and group designations to facilitate
% plotting and identification. Set rarely changes in the Ronne dataset.
% Group can be used to add designations for line width or limit the
% specific set of fractures that are plotted. At a minimum, Group serves to
% group individual features to permit plotting as lines instead of points.
%
% Import data using UI import tool or command line.
%
% Data files are available via FTP in ASCII (.txt) format.
%
% Citing these data:
%    Hulbe, C.L. and C.M. LeDoux. 2011. MOA-derived Structural Feature Map 
%    of the Ronne-Filchner Ice Shelf. Boulder, Colorado USA: National Snow  
%    and Ice Data Center. Digital Media.
%
% An example:

RonneFM_fractures = importdata('RonneFM_fractures.txt');

FMx = RonneFM_fractures(:,3);
FMy = RonneFM_fractures(:,4);
FMset = RonneFM_fractures(:,5);
FMgroup = RonneFM_fractures(:,6);

figure(1)
hold on
sets = unique(FMset);

for s = 1:length(sets)
    
    sind = find(FMset == sets(s));
    groups = unique(FMgroup(sind));
    
    for m = 1:length(groups)
        mind = find(FMgroup==groups(m));
        plot(FMx(mind),FMy(mind),'-');
        
    end
    
    clear mind p
end

axis equal