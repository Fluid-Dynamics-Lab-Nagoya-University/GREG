function [hI, hL] = F_map_plot_sensors(sensi, mask, sensors)

	%display_sensors Template for displaying image data
	%   Displays enso sensors on white w/ black continents
	%    set(gcf,'PaperPositionMode','auto')

    hI = F_map_plot(sensi, mask);

	pivot = sensors;
	sensors_location = zeros(360, 180);
	P = zeros(size(sensi));
    P(pivot) = 1:length(pivot);    
	sensors_location(mask == 1) = P;
	S = reshape(real(sensors_location)', 360 * 180, 1);
	[~, IC, ~] = unique(S);
	
	% align Ilin with pivot somehow
	
	[I, J] = ind2sub(size(sensors_location'), IC(2:end));

	hold on
%     yellow = [246; 191; 0] / 255; % chrome yellow
    green = [0; 110; 79] / 255;
	hL = plot(J, I, 'x', 'MarkerSize', 8, 'LineWidth', 2, ...
        'MarkerFaceColor', 'none', 'MarkerEdgeColor', green);
%     hL.MarkerEdgeColor = yellow;
	hold off

end
