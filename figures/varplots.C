#include "TTree.h"
#include "TRandom3.h"
#include "TMath.h"

TTree* varplots() {
    TTree *tr = new TTree("tr","tr");
    float x,y,a,b;
    tr->Branch("xx",&x,"xx/F");
    tr->Branch("yy",&y,"yy/F");
    tr->Branch("a",&a,"a/F");
    tr->Branch("b",&b,"b/F");

    gRandom = new TRandom3();
    
    for (int i=0; i<100; i++) {
        float phi = gRandom->Uniform(0,TMath::TwoPi());
        a = cos(phi);
        b = sin(phi);
        x = gRandom->Gaus(1,0.5);
        y = gRandom->Gaus(1,0.5);
        tr->Fill();
    }
    tr->SaveAs("tr.root");

    return tr;
}
