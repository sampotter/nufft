function save_as_eps(handle, filename, dpi, width, height);
    set(handle, 'PaperUnits', 'inches', 'PaperPosition', [0 0 width height] / dpi);
    print(handle, '-deps', ['-r' num2str(dpi)], filename);
end