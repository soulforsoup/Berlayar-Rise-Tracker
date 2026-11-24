/* Booking model for the Berlayar Rise tracker.
   A buyer first settles on a stack, then on a floor within it.
   - Stack pull = (how much buyers have favoured the stack so far) x (what is still free in it) ^ LOYALTY.
     LOYALTY 0: buyers keep going to a favoured stack however low its remaining floors are.
     LOYALTY 1: buyers weigh every remaining unit on its own floor, so a stack fades as its high floors go.
   - Within a stack, higher floors go first (FLOOR), with a bonus for the top two floors (TOP2).
   calibrate() replays selection from unit 1 to learn each stack's pull and scores a LOYALTY value against the
   block-by-block snapshots; forecast() plays out the queue ahead of you. */
(function(root){
"use strict";
function mulberry(a){return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function normal(r){var u=1-r(),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}
function gamma(k,r){if(k<1)return gamma(k+1,r)*Math.pow(r(),1/k);var d=k-1/3,c=1/Math.sqrt(9*d);for(;;){var x=normal(r),v=1+c*x;if(v<=0)continue;v=v*v*v;var u=r();if(Math.log(u)<.5*x*x+d-d*v+d*Math.log(v))return d*v}}
function beta(a,b,r){var x=gamma(a,r),y=gamma(b,r);return x/(x+y)}


/* Hybrid floor model.
   We observe WHICH flats have been taken, not the exact sequence of choices.
   Condition on the number taken in each stack, so popular stacks do not
   masquerade as a preference for their particular storey heights.
   Within a stack use a fixed-size weighted-subset likelihood:
      P(booked set | count, floors) proportional to
      exp(floor * sum((storey - 20)/10) + top2 * count(top-two floors)).
   A Gaussian prior keeps small or unusual samples from producing extreme
   weights. Height, views and floor-dependent prices are confounded, so
   these are descriptive preferences, not causal estimates. */
var VERSION='floor-hybrid-v1';
var PAR={floor:Math.log(2),top2:Math.log(1.25)};
var PRIOR_SD={floor:0.65,top2:0.60};
var LEGACY_PAR={floor:1.75,top2:1.5};

function cacheKey(data,state){
  /* Full, deterministic model-input equality: a one-unit swap MUST invalidate
     even when the total booked count and last queue number do not change. */
  return JSON.stringify({
    version:VERSION,
    blocks:data.blocks,
    units:data.units,
    queueLog:state.queueLog,
    blockHistory:state.blockHistory,
    quotaSnap:state.quotaSnap,
    redhill:state.redhill,
    queue:state.meta.queue,
    ethnicity:state.meta.ethnicity,
    targetStacks:state.meta.targetStacks,
    prefs:state.meta.prefs
  });
}
function floorGroups(units){
  var by={},groups=[],booked=0;
  units.forEach(function(u){var key=u.b+'-'+u.s;(by[key]=by[key]||[]).push(u);if(u.t)booked++});
  Object.keys(by).forEach(function(id){
    var us=by[id],k=us.filter(function(u){return u.t}).length;
    if(k===0||k===us.length)return;
    var top=Math.max.apply(null,us.map(function(u){return u.f}));
    groups.push({k:k,items:us.map(function(u){return {
      x:(u.f-20)/10,t:u.f>=top-1?1:0,taken:u.t?1:0
    }})});
  });
  return {groups:groups,booked:booked};
}
function floorMoments(items,k,floor,top2){
  /* Exact elementary-symmetric-polynomial moments of all subsets of size k.
     Descending j prevents double-counting the same flat. */
  var z=new Float64Array(k+1),sx=new Float64Array(k+1),st=new Float64Array(k+1),
      sxx=new Float64Array(k+1),sxt=new Float64Array(k+1),stt=new Float64Array(k+1);
  z[0]=1;
  for(var i=0;i<items.length;i++){
    var x=items[i].x,t=items[i].t,w=Math.exp(floor*x+top2*t);
    for(var j=Math.min(k,i+1);j>=1;j--){
      var z0=z[j-1],x0=sx[j-1],t0=st[j-1];
      z[j]+=w*z0;
      sx[j]+=w*(x0+x*z0);
      st[j]+=w*(t0+t*z0);
      sxx[j]+=w*(sxx[j-1]+2*x*x0+x*x*z0);
      sxt[j]+=w*(sxt[j-1]+x*t0+t*x0+x*t*z0);
      stt[j]+=w*(stt[j-1]+2*t*t0+t*t*z0);
    }
  }
  if(!(z[k]>0)||!isFinite(z[k]))return null;
  var ex=sx[k]/z[k],et=st[k]/z[k];
  return {logz:Math.log(z[k]),ex:ex,et:et,
    xx:Math.max(0,sxx[k]/z[k]-ex*ex),
    xt:sxt[k]/z[k]-ex*et,
    tt:Math.max(0,stt[k]/z[k]-et*et)};
}
function floorObjective(data,floor,top2){
  var pF=1/(PRIOR_SD.floor*PRIOR_SD.floor),pT=1/(PRIOR_SD.top2*PRIOR_SD.top2);
  var ll=-0.5*pF*Math.pow(floor-PAR.floor,2)-0.5*pT*Math.pow(top2-PAR.top2,2);
  var gF=-pF*(floor-PAR.floor),gT=-pT*(top2-PAR.top2);
  var hFF=pF,hFT=0,hTT=pT;
  for(var i=0;i<data.groups.length;i++){
    var group=data.groups[i],m=floorMoments(group.items,group.k,floor,top2);
    if(!m)return null;
    var obsF=0,obsT=0;
    for(var j=0;j<group.items.length;j++)if(group.items[j].taken){
      obsF+=group.items[j].x;obsT+=group.items[j].t;
    }
    ll+=floor*obsF+top2*obsT-m.logz;
    gF+=obsF-m.ex;gT+=obsT-m.et;
    hFF+=m.xx;hFT+=m.xt;hTT+=m.tt;
  }
  return {ll:ll,gF:gF,gT:gT,hFF:hFF,hFT:hFT,hTT:hTT};
}
function learnFloor(units){
  var data=floorGroups(units),f=PAR.floor,t=PAR.top2;
  var fallback={floor:f,top2:t,booked:data.booked,stacks:data.groups.length,method:'prior'};
  if(data.booked<20||data.groups.length<3)return fallback;
  for(var iter=0;iter<20;iter++){
    var o=floorObjective(data,f,t);
    if(!o)return fallback;
    var det=o.hFF*o.hTT-o.hFT*o.hFT;
    if(!(det>1e-9))break;
    var df=(o.hTT*o.gF-o.hFT*o.gT)/det;
    var dt=(o.hFF*o.gT-o.hFT*o.gF)/det;
    var step=1,nf=f,nt=t,trial;
    for(var tries=0;tries<16;tries++){
      nf=Math.max(-0.5,Math.min(2.5,f+step*df));
      nt=Math.max(-1.25,Math.min(1.8,t+step*dt));
      trial=floorObjective(data,nf,nt);
      if(trial&&trial.ll>=o.ll-1e-8)break;
      step*=0.5;
    }
    if(!trial||trial.ll<o.ll-1e-8)break;
    if(Math.abs(nf-f)+Math.abs(nt-t)<1e-6)break;
    f=nf;t=nt;
  }
  if(!isFinite(f)||!isFinite(t))return fallback;
  return {floor:f,top2:t,booked:data.booked,stacks:data.groups.length,method:'hybrid'};
}

function prep(units,par){
  var ids=[],ix={},st=[];
  units.forEach(function(u){var id=u.b+'-'+u.s,k=ix[id];if(k==null){k=ix[id]=ids.length;ids.push(id);st.push({id:id,b:u.b,s:u.s,units:[],top:0,tk:0})}st[k].units.push(u);if(u.f>st[k].top)st[k].top=u.f;if(u.t)st[k].tk++});
  st.forEach(function(s){s.units.sort(function(a,b){return b.f-a.f});s.w=s.units.map(function(u){return Math.exp(par.floor*(u.f-20)/10+(u.f>=s.top-1?par.top2:0))})});
  return {st:st,ix:ix,blocks:st.reduce(function(a,s){if(a.indexOf(s.b)<0)a.push(s.b);return a},[])}
}
/* one pick: returns [stackIndex, unitIndex] or null. free[k][j]=1 if free; sum[k]=sum of free unit weights;
   closed[k]=1 hides a stack from this buyer (block's quota for their group is used up). */
var PW=new Float64Array(64);
function pick(P,a,lam,free,sum,r,closed){
  var K=P.st.length,tot=0,k,j;
  for(k=0;k<K;k++){if(sum[k]>0&&!(closed&&closed[k])){PW[k]=Math.exp(a[k])*Math.pow(sum[k],lam);tot+=PW[k]}else PW[k]=0}
  if(tot<=0)return null;
  var x=r()*tot,last=-1;for(k=0;k<K;k++){if(PW[k]<=0)continue;last=k;x-=PW[k];if(x<=0)break}if(k>=K)k=last;
  var w=P.st[k].w,f=free[k],y=r()*sum[k];last=-1;for(j=0;j<w.length;j++){if(!f[j])continue;last=j;y-=w[j];if(y<=0)break}if(j>=w.length)j=last;
  f[j]=0;var s2=0;for(var i=0;i<w.length;i++)if(f[i])s2+=w[i];sum[k]=s2;return [k,j]
}
/* Replay from an empty project to the current number taken. Learns a[] so the average replay ends with each
   stack's actual count, then reports block counts at each snapshot size. */
function calibrate(units,par,lam,snapSizes,opt){
  opt=opt||{};var P=prep(units,par),K=P.st.length,S=opt.sims||120,rounds=opt.rounds||9,n=units.filter(function(u){return u.t}).length;
  var a=P.st.map(function(s){return Math.log((s.tk+.5)/s.units.length)}),mean,snaps,topFrac;
  for(var rd=0;rd<=rounds;rd++){
    var r=mulberry(opt.seed||11);mean=new Float64Array(K);snaps=snapSizes.map(function(){var o={};P.blocks.forEach(function(b){o[b]=0});return o});var tfN=0,tfD=0;
    for(var s=0;s<S;s++){var free=P.st.map(function(x){return new Uint8Array(x.units.length).fill(1)}),sum=P.st.map(function(x){return x.w.reduce(function(t,v){return t+v},0)}),cnt=new Float64Array(K);
      for(var i=1;i<=n;i++){var p=pick(P,a,lam,free,sum,r);if(!p)break;cnt[p[0]]++;
        for(var q=0;q<snapSizes.length;q++)if(snapSizes[q]===i)for(var k=0;k<K;k++)snaps[q][P.st[k].b]+=cnt[k]/S}
      for(k=0;k<K;k++){mean[k]+=cnt[k]/S;if(cnt[k]>=8){var c=cnt[k],inTop=0;for(var j=0;j<c;j++)if(!free[k][j])inTop++;tfN+=inTop;tfD+=c}}}
    topFrac=tfD?tfN/tfD:null;
    if(rd<rounds)for(k=0;k<K;k++)a[k]+=0.8*Math.log((P.st[k].tk+.3)/(mean[k]+.3));
  }
  return {P:P,a:a,lam:lam,fitMean:Array.prototype.slice.call(mean),snaps:snaps,topFrac:topFrac}
}
/* o: cal (from calibrate), lastQ, queue, drops, called, cShare, cq{block:n}, targets[ids], sims, seed */
function forecast(o){
  var P=o.cal.P,a=o.cal.a,lam=o.cal.lam,K=P.st.length,S=o.sims||600,r=mulberry(o.seed||462),tgt={};(o.targets||[]).forEach(function(id){tgt[id]=1});
  var isT=P.st.map(function(s){return tgt[s.id]?1:0}),blocks=P.blocks,F=[],CL=new Uint8Array(K);for(var f=2;f<=50;f++)F.push(f);
  var surv=P.st.map(function(s){return new Float64Array(s.units.length)}),ok={},run={},ps={},cqEnd={},leftEnd=0,k,j;
  F.forEach(function(f){ok[f]=0;run[f]=[]});blocks.forEach(function(b){cqEnd[b]=0});
  P.st.forEach(function(s,k){if(isT[k]){ps[s.id]={ok:{},top:[]};F.forEach(function(f){ps[s.id].ok[f]=0})}});
  var loc={},rk=[],rkKey=[];P.st.forEach(function(x,k2){x.units.forEach(function(u,j2){loc[u.b+'-'+u.s+'-'+u.f]=[k2,j2]})});
  (o.ranked||[]).forEach(function(key){if(loc[key]){rk.push(loc[key]);rkKey.push(key)}});var pickCnt=new Float64Array(rk.length),noPick=0;
  for(var s=0;s<S;s++){
    var d=beta(o.drops+1,o.called-o.drops+1,r),cq={};blocks.forEach(function(b){cq[b]=o.cq[b]});
    var free=P.st.map(function(x){return Uint8Array.from(x.units,function(u){return u.t?0:1})}),sum=P.st.map(function(x,k2){var t=0;for(var i=0;i<x.w.length;i++)if(free[k2][i])t+=x.w[i];return t});
    var tLeft={},lastTaken={},remaining=0;F.forEach(function(f){tLeft[f]=0});
    for(k=0;k<K;k++)for(j=0;j<free[k].length;j++)if(free[k][j]){remaining++;if(isT[k])for(var f2=2;f2<=P.st[k].units[j].f;f2++)tLeft[f2]++}
    var q=o.lastQ,guard=0,snap=false;
    while(guard++<4000&&(remaining>0||!snap)){
      if(snap&&(tLeft[2]<=0||q>o.queue+900))break;      /* nothing more to learn for the target stacks */
      q++;
      if(q===o.queue&&!snap){snap=true;var best=0;
        for(k=0;k<K;k++){var open=cq[P.st[k].b]>=1,bf=0;for(j=0;j<free[k].length;j++)if(free[k][j]){surv[k][j]++;if(open&&P.st[k].units[j].f>bf)bf=P.st[k].units[j].f}
          if(isT[k]){if(bf>best)best=bf;ps[P.st[k].id].top.push(bf);for(var fi=0;fi<F.length;fi++)if(bf>=F[fi])ps[P.st[k].id].ok[F[fi]]++}}
        F.forEach(function(f){if(best>=f)ok[f]++});blocks.forEach(function(b){cqEnd[b]+=cq[b]});leftEnd+=remaining;
        var got=false;for(var ri=0;ri<rk.length;ri++){var kk=rk[ri][0];if(free[kk][rk[ri][1]]&&cq[P.st[kk].b]>=1){pickCnt[ri]++;got=true;break}}if(!got)noPick++}
      if(remaining<=0)break;
      if(r()<d)continue;
      var chinese=r()<o.cShare,closed=null;
      if(chinese){closed=CL;for(k=0;k<K;k++)CL[k]=cq[P.st[k].b]>=1?0:1}
      var p=pick(P,a,lam,free,sum,r,closed);if(!p)continue;
      remaining--;if(chinese)cq[P.st[p[0]].b]--;
      if(isT[p[0]]){var fl=P.st[p[0]].units[p[1]].f;for(var f3=2;f3<=fl;f3++){tLeft[f3]--;if(tLeft[f3]===0)lastTaken[f3]=q}}
    }
    F.forEach(function(f){if(lastTaken[f]!=null)run[f].push(lastTaken[f])});
  }
  function qtl(x,p){if(!x.length)return null;x=x.slice().sort(function(m,n){return m-n});return x[Math.min(x.length-1,Math.floor(p*x.length))]}
  var out={sims:S,lam:lam,chanceByFloor:{},runOutByFloor:{},perStack:{},cqAtTurn:{},unitsLeftAtTurn:leftEnd/S,survive:{},pick:{},noPick:noPick/S};
  rkKey.forEach(function(key,i){if(pickCnt[i])out.pick[key]=pickCnt[i]/S});
  F.forEach(function(f){out.chanceByFloor[f]=ok[f]/S;out.runOutByFloor[f]={p10:qtl(run[f],.1),p50:qtl(run[f],.5),p90:qtl(run[f],.9)}});
  for(var id in ps){out.perStack[id]={chanceByFloor:{},topMedian:qtl(ps[id].top,.5),topP90:qtl(ps[id].top,.9)};F.forEach(function(f){out.perStack[id].chanceByFloor[f]=ps[id].ok[f]/S})}
  blocks.forEach(function(b){out.cqAtTurn[b]=cqEnd[b]/S});
  P.st.forEach(function(x,k2){x.units.forEach(function(u,j2){out.survive[u.b+'-'+u.s+'-'+u.f]=u.t?0:surv[k2][j2]/S})});
  return out}
var LAMS=[0,0.25,0.5,0.75,1];
/* ctx: units, lastQ, queue, drops, cShare, cq, targets, snaps:[{n, taken:{block:count}}] (earlier block snapshots) */
function runLambda(ctx,lam,opt){
  opt=opt||{};var sizes=ctx.snaps.map(function(s){return s.n});
  var cal=calibrate(ctx.units,ctx.par||learnFloor(ctx.units),lam,sizes,{sims:opt.calSims||60,rounds:opt.rounds||8});
  var loss=0;ctx.snaps.forEach(function(s,i){for(var b in s.taken){var m=cal.snaps[i][b]||0;loss+=(m-s.taken[b])*(m-s.taken[b])/(m+1)}});
  var fc=forecast({cal:cal,lastQ:ctx.lastQ,queue:ctx.queue,drops:ctx.drops,called:ctx.lastQ,cShare:ctx.cShare,cq:ctx.cq,targets:ctx.targets,ranked:ctx.ranked,sims:opt.sims||250,seed:462+Math.round(lam*100)});
  return {lam:lam,loss:loss,fc:fc}
}
/* Average the runs, weighting each LOYALTY value by how well its replay matched the earlier snapshots. */
function combine(runs){
  var best=Math.min.apply(null,runs.map(function(x){return x.loss})),w=runs.map(function(x){return Math.exp(-(x.loss-best)/2)}),W=w.reduce(function(a,b){return a+b},0);
  w=w.map(function(x){return x/W});
  function avg(get){var t=0;runs.forEach(function(x,i){t+=w[i]*get(x.fc)});return t}
  function rng(get){var v=runs.map(function(x){return get(x.fc)});return [Math.min.apply(null,v),Math.max.apply(null,v)]}
  var o={weights:runs.map(function(x,i){return {lam:x.lam,w:w[i],loss:x.loss}}),chance:{},range:{},runOut:{},perStack:{},cqAtTurn:{},survive:{}},f0=runs[0].fc;
  Object.keys(f0.chanceByFloor).forEach(function(f){o.chance[f]=avg(function(c){return c.chanceByFloor[f]});o.range[f]=rng(function(c){return c.chanceByFloor[f]});
    var ro=runs.map(function(x){return x.fc.runOutByFloor[f]}).filter(function(v){return v.p50!=null});
    o.runOut[f]=ro.length===runs.length?{lo:Math.min.apply(null,ro.map(function(v){return v.p10})),mid:Math.round(avg(function(c){return c.runOutByFloor[f].p50})),hi:Math.max.apply(null,ro.map(function(v){return v.p90}))}:null});
  Object.keys(f0.perStack).forEach(function(id){o.perStack[id]={chance:{},top:Math.round(avg(function(c){return c.perStack[id].topMedian}))};Object.keys(f0.chanceByFloor).forEach(function(f){o.perStack[id].chance[f]=avg(function(c){return c.perStack[id].chanceByFloor[f]})})});
  Object.keys(f0.cqAtTurn).forEach(function(b){o.cqAtTurn[b]=avg(function(c){return c.cqAtTurn[b]})});
  Object.keys(f0.survive).forEach(function(u){o.survive[u]=Math.round(100*avg(function(c){return c.survive[u]}))/100});
  o.pick={};runs.forEach(function(x,i){for(var key in x.fc.pick)o.pick[key]=(o.pick[key]||0)+w[i]*x.fc.pick[key]});
  for(var pk in o.pick)o.pick[pk]=Math.round(1000*o.pick[pk])/1000;
  return o}
/* Order every unit the way the buyer would pick. pf: budget, groups[{id,label}], steps[...]. Returns [{key,u,grp,sub}];
   units over budget are left out. */
function rank(units,pf){
  var by={},out=[],seen={};units.forEach(function(u){var id=u.b+'-'+u.s;(by[id]=by[id]||[]).push(u)});
  /* step: {grp, stacks[ids], min, max, floors[listed floors come first, the rest follow from the top down], pooled} */
  (pf.steps||[]).forEach(function(st){var us=[],lo=st.min||2,hi=st.max||99,fl=st.floors||[];
    st.stacks.forEach(function(id,i){(by[id]||[]).forEach(function(u){var key=u.b+'-'+u.s+'-'+u.f;if(u.f>=lo&&u.f<=hi&&u.p<=pf.budget&&!seen[key]){var k=fl.indexOf(u.f);us.push([u,i,k>=0?k:1000-u.f,key])}})});
    us.sort(st.pooled?function(a,b){return a[2]-b[2]||a[1]-b[1]}:function(a,b){return a[1]-b[1]||a[2]-b[2]});
    us.forEach(function(x){seen[x[3]]=1;out.push({key:x[3],u:x[0],grp:st.grp,sub:x[0].b+' · '+x[0].s})})});
  return out}
root.BRModel={VERSION:VERSION,rank:rank,prep:prep,calibrate:calibrate,forecast:forecast,runLambda:runLambda,combine:combine,learnFloor:learnFloor,cacheKey:cacheKey,LAMS:LAMS,PAR:PAR,LEGACY_PAR:LEGACY_PAR,mulberry:mulberry};
})(typeof window!=='undefined'?window:globalThis);
