// Dependency-free contract tests for all feature endpoints and large JSON multipart files.
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const { execFileSync } = require('child_process');
let ts;
try { ts = require('typescript'); }
catch { ts = require(path.join(execFileSync('npm', ['root', '-g'], {encoding:'utf8'}).trim(), 'typescript')); }
const source = fs.readFileSync(path.join(__dirname, '../src/app/feature-lab.service.ts'), 'utf8');
const output = ts.transpileModule(source, {
  compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, experimentalDecorators:true}
}).outputText;
const calls = [];
const http = {
  get: url => {calls.push({route:url.replace('http://localhost:8000',''),fields:null});return Promise.resolve(new Blob(['PNG']));},
  post: (url,body,options) => {
    const fields = Object.fromEntries(body.entries());
    calls.push({route:url.replace('http://localhost:8000',''), fields, options});
    if (options?.observe === 'response') return Promise.resolve({body:new Blob(['PNG']),headers:{get:()=>'1'}});
    return Promise.resolve(options?.responseType === 'blob' ? new Blob(['PNG']) : {keypoints:[],descriptors:[],matches:[]});
  }
};
const exported = {};
const context = {
  exports:exported,module:{exports:exported},console,Blob,File,FormData,
  require:id=>{
    if (id === '@angular/core') return {Injectable:()=>cls=>cls};
    if (id === 'rxjs') return {firstValueFrom:promise=>promise};
    throw Error('Unmocked import '+id);
  }
};
vm.runInNewContext(output,context);
(async()=>{
  const s=new exported.FeatureLabService(http);
  const a=new Blob(['a']), b=new Blob(['b']);
  const feature={keypoints:[[10,20]],descriptors:[[1,2]],keypoint_details:[],num_keypoints:1,execution_time_ms:2};
  const corners={method:'harris',descriptor_method:'mops',threshold:0.005,window_size:5,margin:null,max_keypoints:null};
  await s.getLatestImage();
  await s.uploadLatestImage(a);
  await s.extractCorners(a,corners);
  await s.cornerOverlay(a,corners,5);
  await s.extractSift(a,{max_keypoints:null, contrast_threshold:.03,edge_threshold:10,num_octaves:3,num_intervals:3,sigma:1.6,assumed_blur:.5,num_cells:4,num_bins:8,peak_ratio:.8,descriptor_threshold:.2});
  await s.siftOverlay(a,{max_keypoints:null},null,true);
  const settings={method:'harris',metric:'ssd',ratio:.8,threshold:null,cross_check:true,patch_size:11,max_matches:null};
  await s.match(a,b,feature,feature,settings);
  await s.visualizeMatches(a,b,feature,feature,settings,6,true);
  const expected=['/images/latest','/images/upload','/corners','/corners/overlay','/features/sift','/features/sift/visualize','/features/match','/features/match/visualize'];
  if (JSON.stringify(calls.map(c=>c.route)) !== JSON.stringify(expected)) throw Error('Unexpected API calls');
  if (calls[2].fields.max_keypoints!==undefined || calls[4].fields.max_keypoints!==undefined) throw Error('Extraction cap should not be sent');
  if (calls[6].fields.max_matches!==undefined) throw Error('Matching cap should not be sent');
  for (const c of calls.slice(6)) {
    const { features1, features2 } = c.fields;
    if (!(features1 instanceof Blob) || !(features2 instanceof Blob)) throw Error('Descriptor JSON must be multipart FILE parts');
    if (features1.name !== 'features1.json' || features2.name !== 'features2.json') throw Error('Multipart JSON filenames not provided');
    if (features1.type !== 'application/json') throw Error('Wrong descriptor content type');
    if (!c.fields.image1 || !c.fields.image2) throw Error('Image parts missing');
    const d1=JSON.parse(await features1.text()), d2=JSON.parse(await features2.text());
    if (d1.keypoints[0][0]!==10 || d2.descriptors[0][1]!==2) throw Error('Incorrect descriptor JSON');
  }
  const big = {keypoints:[],descriptors:[]};
  for(let i=0;i<6000;i++){
    big.keypoints.push([i,i]);
    big.descriptors.push(Array.from({length:64}, (_,j)=>Math.sin(i+j)));
  }
  await s.match(a,b,big,feature,settings);
  const bigJsonFile=calls.at(-1).fields.features1;
  if (!(bigJsonFile instanceof Blob) || bigJsonFile.size < 1024*1024) throw Error('Large JSON multipart test failed');
  console.log('PASS: all 8 API contracts, unlimited count, JSON upload files >1 MiB');
})().catch(e=>{console.error(e);process.exitCode=1});
