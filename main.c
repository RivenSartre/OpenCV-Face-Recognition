/**************************************************************************************
*		              电子密码锁												  *
实现现象：下载程序输入正确密码，会听到继电器吸合的声音
硬件连接：
					 P1 -->矩阵键盘
					 P20--> EEPROM模块SDA
					 P21--> EEPROM模块SCL
					 P22-->继电器模块J15
					 
					 矩阵键盘：
					 0      1      2       3
					 
					 4      5      6       7
					 
					 8      9    返回    确定
					 
					 A      B      C       D
注意事项：无																				  
***************************************************************************************/

#include "reg52.h"		       //此文件中定义了单片机的一些特殊功能寄存器
#include "lcd.h"
#include "key.h"
#include "i2c.h"


#define u16 unsigned int 	  //对数据类型进行声明定义
#define u8 unsigned char

sbit AS=P1^0;               //继电器  *******

sbit conn1=P2^1 ;			 //****
sbit conn2=P2^4 ;			 //****
sbit conn3=P2^3 ;
sbit conn4=P2^2;			  //****

/*****************************************************************************/

/*****************************************************************************/
u8 pw_num,Error_Num,PassWord_Length=6;
u8 PASSWORD[]={8,8,8,8,8,8,0,0,0,0};
u8 INPUT_PW_Tab[10];
u8 key_num,Step,Step5,Load_first_flag=0;
/*****************************************************************************/

/*******************************************************************************/
bit result_flag,Input_suc_flag;
bit List1=0; 
/*****************************************************************************/

/*******************************************************************************/
void Step_0();
void Step_1();
void Step_2();
void Step_3();
void Step_4();
	  
void Step_5();
	   void Step5_0();
	   void Step5_1();
	   void Step5_2();
	   void Step5_3();
	   void Step5_4();
	   void Step5_5();
void Step_6();
void Step_7();						//****
void Step_8();
void Step_9();
void Step_10();							//****
void Int0Init();					 //****

/********************************************************************************/
   

/*******************************************************************************/
void CipherComparison();
void input_password(bit m);
void Read_Password();

/*******************************************************************************
* 函 数 名       : main
* 函数功能		 : 主函数
* 输    入       : 无
* 输    出    	 : 无
*******************************************************************************/
void main()
{	
	u8 data1,a;
	//	ShowString(0x00," R&C face-recognition door ");  
	//	At24c02Write(0,0);                                                //开机显示密码
	LcdWriteCom(0x01);  //清屏
	for(data1=0;data1<PassWord_Length+2;data1++)
	{
		a=At24c02Read(data1)+0x30;
	 	LcdWriteData(a);	
		delay(1000);	
    }
	delay(1000);
	LcdInit();
	delay(1000);
	Step=0;
	Step5=0;
	Error_Num=0x00;
    Read_Password();	
						   //****
	Int0Init();  //	设置外部中断0	****
	conn2=0;				 //****
	conn1=0;				//****
	conn4=0;
	while(1)
				{				AS=1;
								key_num=KeyDown();                   //读取输入值
										switch(Step)
										{
												case 0:
												 {Step_0();break;}
												case 1:
												 {Step_1();break;}
												case 2:
												 {Step_2();break;}
												case 3:
												 {Step_3();break;}	
												case 4:
												 {Step_4();break;}	
												case 5:
												 {Step_5();break;}	
												case 6:
												 {Step_6();break;}
												case 7:
												 {Step_7();break;} //按下黄键 
												case 8:
												 {Step_8();break;} //按下蓝键
												 case 9:
												 {Step_9();break;} //按下蓝键
												 case 10:
												  {Step_10();break;}
										}
				 }				
}

/*****************************************************************************/

/*******************************************************************************
* 函 数 名         : Int1Init()
* 函数功能		   : 设置外部中断1
* 输    入         : 无
* 输    出         : 无
*******************************************************************************/
void Int0Init()					   //****************
{
	//设置INT0
	IT0=1;//跳变沿出发方式（下降沿）
	EX0=1;//打开INT0的中断允许。	
	EA=1;//打开总中断	
}


void Int0()	interrupt 0		//外部中断0的中断函数	  //****************
{
	delay(1000);	 //延时消抖
	if(key_num==0x0f)
	{
		AS=1;
		Step=0;
	}
}


/*******************************************************************************/
void Step_0()				   //********************
{
		LcdInit();
		ShowString(0x00,"face-recognition"); //****                          //第一行显示Pechin Science 
	    ShowString(0x10,"    WELCOME!     ");            ///*****               // 第二行显示 WELCOME!  
		while(KeyDown()==0xff)Step=9;
		
}
/*****************************************************************************/

void Step_9()
{
	 LcdInit();
	ShowString(0x00," A:New B:facial "); //****                          //第一行显示Pechin Science 
	  ShowString(0x10," C;Key D:close ");
	  while(KeyDown()==0xff)Step=10;
}

void Step_10()
{

	if(key_num==0x0c) {Step=7;}
	else  if(key_num==0x0d) {Step=8;}
	else if(key_num==0x0e) {Step=1;}
	else if(key_num==0x0f) {AS=1;Step=0;}	
	else Step=10;
}

/*******************************************************************************/
void Step_1()
{
		LcdWriteCom(0x01);                                           //清屏
	  ShowString(0x00,"Unlock");
		ShowString(0x0f,"<");	                                       //1602第一行显示unlock
	  ShowString(0x10,"Change Password");
		ShowString(0x1f," "); 	                                       // 1602第二行显示Change Password                                                                            	
	  Step=2;	                                                      //
}
/*****************************************************************************/

/*******************************************************************************/
void Step_2()
{
	if(key_num!=0x0b)  
		{
					if((key_num==0x01) ||( key_num==0x09))                    //1键或9键按下
					{
								List1=~List1;                           //Change Password
								if(List1==0)
									{                                                  
									ShowString(0x0f,"<");                           // Unlock            <
									ShowString(0x1f," ");	                          // Change Password  
								}
								else
								{
									ShowString(0x0f," ");                           // Unlock            
									ShowString(0x1f,"<");	                          // Change Password   <
								}
					}
		}
	else                                                           //确认键按下
	{
					if(List1==0){Step=3;}
					else        {Step=5;List1=0;}
	}
}
/*****************************************************************************/

/*******************************************************************************/
void Step_3()                                             //
{
				Step=4;
	            pw_num=0;
				LcdInit();
				ShowString(0x00,"Pass Word:  ");
}
/*****************************************************************************/

/*******************************************************************************/
void Step_4()                                                 
{
	   input_password(0);                                              //输入密码并以*显示
	   if(Input_suc_flag==1){Step=6;}                                  //密码输入完成进入下一步
      Input_suc_flag=0;                                              //清除密码输入完成标志
}
/*****************************************************************************/

/*******************************************************************************/
void Step_5()                                                        //修改密码
{
	switch(Step5)
	{
			case 0: {Step5_0();}  break;
			case 1: {Step5_1();}  break;
			case 2: {Step5_2();}  break;
			case 3: {Step5_3();}  break;
		    case 4: {Step5_4();}  break;
			case 5:	{Step5_5();}	break;
	}
}
/*****************************************************************************/

/*******************************************************************************/
void Step_6()
{
	u8 i=0;
	CipherComparison();                                //密码比对
	if(result_flag==1)                                 //密码正确
		{
				LcdInit();
				ShowString(0x00,"    WELCOME!");
				AS=0;                                         //开继电器
				delay(200000);
				
	
				for(i=0;i<60;i++)	  //循环100次，也就是大约5S
				{
					AS=0;			 //开启电机
					delay(5000);	//大约延时50ms
				}
				
				AS=1;	                                        //关继电器
		}
	else                                                //密码错误
		{
				LcdInit();
				ShowString(0x00,"Error 01!");
		}
		Step=0;
}

/****************************************************************************************
密码输入函数
密码长度：pw_num
密码输入完成标志：Input_suc_flag
*****************************************************************************************/

void Step5_0()                 
{
	  LcdWriteCom(0x01);  //清屏
	  ShowString (0x00,"Input PassWord:");     //1602显示：输入密码
	  Step5=1;
	  pw_num=0;
}
void Step5_1()                
{
		input_password(0);                  //   输入密码并以*显示
		if(Input_suc_flag==1)               //密码输入完成
			{
							Step5=2;                  //
							Input_suc_flag=0;         //清除密码输入完成标志
			}
}
void Step5_2()                //
{
     CipherComparison();                //密码比对
	   Step5=3;
}
void Step5_3()                //
{
    if(result_flag==0)         //        密码错误
				{
							if(Error_Num<3)             //输出错误次数小于3
							{
								Error_Num++;
								LcdInit();
								ShowString (0x00,"Error 01");
								delay(20000);
								Step5=0;
							}
							else                          //密码错误次数大于3
							{
								Error_Num=0;
								Step=0;
							}					
				}
		else						      					       //密码正确
				{
					LcdInit();
					ShowString (0x00,"New PassWord:");	
					pw_num=0;
					Step5=4;
				}  					
}
void Step5_4()
{
		input_password(1);     					       //输入密码并显示
		if(Input_suc_flag==1)                  //输入完成
			{ 
				Step5=5;
				Input_suc_flag=0;
				LcdWriteCom(0x01);  //清屏
	      ShowString (0x00,"      OK!");
			}
}
void Step5_5()
{
	unsigned char j;   	
	PassWord_Length=pw_num;                         //读取输入密码长度
	At24c02Write(0,Load_first_flag);        
  delay(100);	
	At24c02Write(1,PassWord_Length);                //保存 密码长度
	delay(100);
	for(j=0;j<PassWord_Length;j++)         
	{
		PASSWORD[j]=INPUT_PW_Tab[j];               		 //读取密码
		At24c02Write(j+2,INPUT_PW_Tab[j]);         		 //保存密码至EEPROM
		delay(100);
}
	Step5=0;
	Step=0;
}


void Read_Password()
{
	unsigned char j;
	Load_first_flag=At24c02Read(0x00);
//	if(Load_first_flag==0)         		 //初次运行  初始密码错误可以将此句打开重新编译下载
	{
		Load_first_flag=1;
		At24c02Write(0,0x01);
		delay(100);
		At24c02Write(1,0x06);           //写默认密码长度6至EEPROM
		delay(100);
		for(j=0;j<PassWord_Length;j++)
		{
			At24c02Write(j+2,8);              //写默认密码888888至EEPROM
			PASSWORD[j]=INPUT_PW_Tab[j];     //读密码
			delay(100);
		}
	}	
	Load_first_flag=At24c02Read(0x00);
	PassWord_Length=At24c02Read(0x01);  //读取密码长度
	for(j=0;j<PassWord_Length;j++)       //读取密码
	{
		PASSWORD[j]=At24c02Read(j+2);
	}
	
}
void input_password(bit m)
{
	unsigned char j;
	if(key_num!=0x0b)                                      //ok键没有按下
		{
					if(key_num<0x0a)                               //1-9按下
					{
						INPUT_PW_Tab[pw_num]=key_num;                //保存至输入密码数组
						pw_num=pw_num+1;                              //密码长度+1
						 LcdWriteCom(0xc0);
						 for(j=0;j<pw_num;j++)
								{
												if(m==0) {LcdWriteData('*');	}                        //密码隐藏
												else     {LcdWriteData(INPUT_PW_Tab[j]+0x30);}         //显示密码
								}
											
					}
					if(key_num==0x0a)                              //返回键按下
					{
						if(pw_num!=0) {pw_num=pw_num-1;}
						else          {Step=0;}
						// ShowString (0x00,"Pass Word：");  
                    LcdWriteCom(0xc0);						
										for(j=0;j<pw_num;j++)
											{
														if(m==0) {LcdWriteData('*');	}                   //密码隐藏
														else     {LcdWriteData(INPUT_PW_Tab[j]+0x30);}    //显示密码	
										   }
											LcdWriteData(' '); 
					}
		} 
	else                          //ok键按下
	  {
			if(pw_num==0)	  
					{
							Step=0;
						  LcdWriteCom(0x01);
						  ShowString (0x00,"Error 02!");
							delay(10000);
					}
			else{		
							Input_suc_flag=1; 
					}
					 //AS=0;			
		}                
}
/*******************************************************
                密码比对
********************************************************/
void CipherComparison()
{ 
	u8 i,j=0;
	if(PassWord_Length==pw_num)                       //密码长度比对
	{
		for(i=0;i<PassWord_Length;i++)                  //密码比对
		{	
				if(PASSWORD[i]!=INPUT_PW_Tab[i])
					{
						result_flag=0;break;                     //密码错误
					}
				else
					{
						result_flag=1;                          //密码正确
					}
					  INPUT_PW_Tab[i]=0XFF;                   //清除密码缓存数组
		}
	}
	else
	{result_flag=0;}	
}





void Step_7()					//*******
{
	u8 i=0;
	LcdWriteCom(0x01);  //清屏
	conn1=1;
	conn2=0;
	LcdInit();
	ShowString (0x00,"Recognitng...");     //1602显示：正在识别
	while(1)
	{
		key_num=KeyDown(); 
		
		
		
		if(conn3==0) {
			Step=0;
			ShowString (0x10,"complete!");     //1602显示：识别成功
			delay(10000);
			conn1=0;
			conn2=0;	
			Step=0;
			break;
		}

		else if(key_num==0x0b)
		{
				ShowString(0x10,"    Return!");
				AS=1;
				conn4=1;
				for(i=0;i<60;i++)	  //循环100次，也就是大约5S
				{
								 //开启电机
					delay(5000);	//大约延时50ms
				}
				conn1=0;
				conn2=0;
				Step=0;
				conn4=0;
				break;
		}
			
	}
}


void Step_8()						   //******
{
	u8 i=0;
	//u16 temp=2001;
	LcdWriteCom(0x01);  //清屏
	
	
	ShowString (0x00,"look the camera");     //1602显示：正在识别

	conn1=0;
	conn2=1;
	while(1)
	{
			
		
		key_num=KeyDown(); 
		if(conn3==0)                                 //密码正确
		{				
				ShowString(0x10,"    WELCOME!");
				
				AS=0;                                         //开继电器
				for(i=0;i<100;i++)	  //循环100次，也就是大约5S
				{
					AS=0;			 //开启电机
					delay(5000);	//大约延时50ms
				}
				AS=1;
				delay(60000);
					                                        //关继电器
			
		}
		
		else if(key_num==0x0b)
		{
				ShowString(0x10,"    Return!");
				AS=1;
				conn4=1;
				
				conn1=0;
				conn2=0;
				for(i=0;i<60;i++)	  //循环100次，也就是大约5S
				{
								 //开启电机
					delay(5000);	//大约延时50ms
				}
				Step=0;
				conn4=0;
				break;
		}	
	}
	Step=0;
}

