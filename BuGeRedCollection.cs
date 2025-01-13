using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing.Drawing2D;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Drawing.Interop;
using System.Numerics;

namespace Phi3
{
    internal class BuGeRedCollection
    {
        public List<BuGeRed> GetBuGeRedListFromBitmap(Bitmap sourceImage)
        {
            bool isfirstpixel = true;
            BuGeRed? firstpixel;


            BitmapData sourceData = sourceImage.LockBits(new Rectangle(0, 0,
                        sourceImage.Width, sourceImage.Height),
                        ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);


            byte[] sourceBuffer = new byte[sourceData.Stride * sourceData.Height];
            //TODO: what to do with cloud and Linux..
            Marshal.Copy(sourceData.Scan0, sourceBuffer, 0, sourceBuffer.Length);
            sourceImage.UnlockBits(sourceData);

            List<BuGeRed> pixelList = new List<BuGeRed>(sourceBuffer.Length / 4);

            using (MemoryStream memoryStream = new MemoryStream(sourceBuffer))
            {
                memoryStream.Position = 0;
                BinaryReader binaryReader = new BinaryReader(memoryStream);

                while (memoryStream.Position + 4 <= memoryStream.Length)
                {
                    if (isfirstpixel && memoryStream.Position == 0)
                    {
                        firstpixel = new BuGeRed(binaryReader.ReadBytes(4));
                        //firstpixel.BGRDiff.BasePixel = firstpixel;
                        pixelList.Add(firstpixel);
                        isfirstpixel = false;
                    }
                    else
                    {
                        BuGeRed pixel = new BuGeRed(binaryReader.ReadBytes(4));
                        //pixel.BGRDiff.BasePixel = firstpixel ?? new BuGeRed(Color.Black);
                        pixelList.Add(pixel);
                    }

                }
                binaryReader.Close();
            }
            return pixelList;
        }


        public byte[] GetBytesFromBitmap(Bitmap sourceImage)
        {

            BitmapData sourceData = sourceImage.LockBits(new Rectangle(0, 0,
                        sourceImage.Width, sourceImage.Height),
                        ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);


            byte[] sourceBuffer = new byte[sourceData.Stride * sourceData.Height];
            //TODO: what to do with cloud and Linux..
            Marshal.Copy(sourceData.Scan0, sourceBuffer, 0, sourceBuffer.Length);
            sourceImage.UnlockBits(sourceData);

            return sourceBuffer;

        }


        public byte[] GetBytesFromBuGeRedList(List<BuGeRed> pixelList, int width, int height)
        {
            Bitmap resultBitmap = new Bitmap(width, height, PixelFormat.Format32bppArgb);

            BitmapData resultData = resultBitmap.LockBits(new Rectangle(0, 0,
                        resultBitmap.Width, resultBitmap.Height),
                        ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

            byte[] resultBuffer = new byte[resultData.Stride * resultData.Height];


            Marshal.Copy(resultData.Scan0, resultBuffer, 0, resultBuffer.Length);

            //Marshal.Copy(resultBuffer, 0, resultData.Scan0, resultBuffer.Length);
            //resultBitmap.UnlockBits(resultData);

            return resultBuffer;
        }


        public Bitmap GetBitmapFromBuGeRedList(List<BuGeRed> pixelList, int width, int height)
        {
            Bitmap resultBitmap = new Bitmap(width, height, PixelFormat.Format32bppArgb);

            BitmapData resultData = resultBitmap.LockBits(new Rectangle(0, 0,
                        resultBitmap.Width, resultBitmap.Height),
                        ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

            byte[] resultBuffer = new byte[resultData.Stride * resultData.Height];

            using (MemoryStream memoryStream = new MemoryStream(resultBuffer))
            {
                memoryStream.Position = 0;
                BinaryWriter binaryWriter = new BinaryWriter(memoryStream);

                foreach (BuGeRed pixel in pixelList)
                {
                    binaryWriter.Write(pixel.GetBytes());
                }

                binaryWriter.Close();
            }

            Marshal.Copy(resultBuffer, 0, resultData.Scan0, resultBuffer.Length);
            resultBitmap.UnlockBits(resultData);

            return resultBitmap;
        }
        

        public List<Vector3> GetBitmapAsVector3(Bitmap sourceImage)
        {
            
            List<BuGeRed> l = GetBuGeRedListFromBitmap(sourceImage);
            List<Vector3> vl = new List<Vector3>(l.Count);

            foreach (var bgr in l)
            {
                vl.Add(bgr.GetAsPercentageVector3());
            }
            return vl;
        }

    }
}
