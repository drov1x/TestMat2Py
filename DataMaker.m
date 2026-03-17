function DataMaker()
    a = 1;
    b = 10.111111111111111111111;
    c = a + b;
    d = a - b;
    save("input.mat", "a", "b");
    save("output.mat", "c", "d");
end