function hI = F_map_time_plot(sensi, mask, time)

	%display_sensors Template for displaying image data
	%   Displays enso sensors on white w/ black continents
	%    set(gcf,'PaperPositionMode','auto')

    hI = F_map_plot(sensi, mask);

	day_name = datenum('1800/01/01', 'yyyy/mm/dd') + time;
	g = datestr(day_name, 'yyyy/mm/dd');
    hold on
	title(g, 'FontSize', 20, 'FontName', 'Times New Roman');
    hold off

end