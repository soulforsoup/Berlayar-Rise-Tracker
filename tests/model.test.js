"use strict";
const assert=require('node:assert/strict');
require('../model.js');
const M=globalThis.BRModel;
assert.equal(M.VERSION,'floor-hybrid-v1');
function sample(type){
  const us=[];
  for(let s=0;s<6;s++)for(let f=2;f<=13;f++){
    us.push({b:'X',s:String(s),f:f,p:100000+f*100,t:type==='high'?+(f>=10):+(f<=5)});
  }
  return us;
}
const high=M.learnFloor(sample('high')),low=M.learnFloor(sample('low'));
assert.ok(high.floor>low.floor,'learnt slope should respond to observed choices');
assert.ok(high.booked>=20);
assert.equal(M.learnFloor(sample('high').map(u=>({...u,t:0}))).method,'prior');
const base={blocks:[],units:[{b:'X',s:'1',f:20,p:100,t:1},{b:'X',s:'1',f:21,p:110,t:0}]};
const state={queueLog:[{d:'2026-11-23',last:255,taken:1}],blockHistory:[{d:'2026-11-23',taken:{X:1}}],quotaSnap:{left:{X:{C:100}}},redhill:{finalDropout:.14},meta:{queue:462,ethnicity:'Chinese',targetStacks:['X-1'],prefs:{budget:751600}}};
const a=M.cacheKey(base,state);
function clone(x){return JSON.parse(JSON.stringify(x))}
let b=clone(base);b.units[0].t=0;b.units[1].t=1;
assert.notEqual(a,M.cacheKey(b,state),'swapping booked flats must invalidate cache');
let q=clone(state);q.queueLog[0].last++;assert.notEqual(a,M.cacheKey(base,q));
q=clone(state);q.blockHistory[0].taken.X++;assert.notEqual(a,M.cacheKey(base,q));
q=clone(state);q.quotaSnap.left.X.C--;assert.notEqual(a,M.cacheKey(base,q));
q=clone(state);q.meta.prefs.budget++;assert.notEqual(a,M.cacheKey(base,q));
assert.equal(a,M.cacheKey(clone(base),clone(state)),'unchanged inputs must retain their cache key');
console.log('Hybrid floor learning and cache invalidation tests passed.');
