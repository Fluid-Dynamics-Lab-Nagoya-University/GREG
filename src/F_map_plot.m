function hI = F_map_plot(sensi, mask)

	%display_sensors Template for displaying image data
	%   Displays enso sensors on white w/ black continents
	%    set(gcf,'PaperPositionMode','auto')

    figure
	snapshot = NaN * zeros(360 * 180, 1);
	x = sensi';
	snapshot(mask == 1) = x;
	C = reshape(real(snapshot), 360, 180)';
	
    hI = imagesc(C);
	shading interp
% 	jetmod = jet(256);
% 	jetmod(1, :) = 0;
% 	colormap(jetmod);
    Nmap = 50;
    linearvar = (0:Nmap-1)' / Nmap;
    alinearvar = flipud(linearvar);
    map = [[0; linearvar; 1; ones(Nmap, 1)], ...
        [0; linearvar; 1; alinearvar], ...
        [0; ones(Nmap, 1); 1; alinearvar]];
	colormap(map);
% 	colorbar
	caxis([-10 10]);
	set(gca, 'FontName', 'Times', 'Color', 'white', 'FontSize', 20);

%     axis equal
	axis off
    pbaspect([2 1 1])
    set(gcf,'PaperPositionMode','auto')

end