// Next.js API route support: https://nextjs.org/docs/api-routes/introduction

//Normally this would be some secure API
export default function keyHandler(req, res) {
  return res.json({
    key: process.env.PUBLIC_MAPBOX_CREDENTIALS,
    status: 200,
  });
}
