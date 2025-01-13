using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
//using System.Drawing.Imaging;
using System.Numerics;

namespace Phi3
{


    public record BuGeRed
    {


        public byte Blue;
        public byte Green;
        public byte Red;
        public byte Alpha;
        //public BGRDiff BGRDiff;


        public BuGeRed(int v, byte[] colorsWithAlpha)
        {
            Blue = colorsWithAlpha[0];
            Green = colorsWithAlpha[1];
            Red = colorsWithAlpha[2];
            Alpha = colorsWithAlpha[3];
        }


        public BuGeRed(Color c)
        {
            Blue = c.B;
            Green = c.G;
            Red = c.R;
            Alpha = c.A;
        }


        public BuGeRed(BuGeRed copyMe)
        {

            this.Blue = copyMe.Blue; this.Green = copyMe.Green;
            this.Red = copyMe.Red; this.Alpha = copyMe.Alpha;
        }


        public byte[] GetBytes()
        {
            return new byte[] { this.Blue, this.Green, this.Red, this.Alpha };
        }

        public Vector3 GetAsVector3()
        {
            return new Vector3((float)this.Blue, (float)this.Green, (float)this.Red);
        }

        public Vector3 GetAsPercentageVector3()
        {
            return new Vector3((float)this.Blue / 255.0f, (float)this.Green / 255.0f, (float)this.Red / 255.0f);
        }


        public override int GetHashCode()
        {
            //TODO: Change to b+g+r+a  - what happens if uint is outside int - may happen and as this is a hash:
            // Might have odd behaviour as identical things are not regognised as such.
            if (BitConverter.IsLittleEndian)
                Array.Reverse(this.GetBytes());

            return BitConverter.ToInt32(this.GetBytes(), 0);

            //return (int)this.Blue;
        }


        public BuGeRed(byte[] barr)
        {


            this.Blue = barr[0];
            Green = barr[1];
            Red = barr[2];
            Alpha = barr[3];

        }




    }
}
