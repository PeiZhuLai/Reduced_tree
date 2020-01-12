void testOpenRemoteFile(){
	TFile* f = TFile::Open("root://cmsio5.rc.ufl.edu//store/user/t2/users/rosedj1/ForPeeps/ForChenguang/crab_DoubleEG_Run2016BCDEFGH-03Feb2017_fullyhadded.root");
	TTree* t = (TTree*)f->Get("Ana/passedEvents");
	t->Print();
}
